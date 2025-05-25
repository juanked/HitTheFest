import os
import sys
import json
import time
import click
import numpy as np
from tqdm import tqdm
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import unicodedata

# Load .env for Spotify credentials
load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIPY_REDIRECT_URI')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def print_debug(msg, debug):
    if debug:
        click.echo(f"[DEBUG] {msg}")

def validate_artists_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False, "JSON root is not an object"
        if "artists" not in data:
            return False, '"artists" key not found'
        if not isinstance(data["artists"], list):
            return False, '"artists" is not a list'
        if not all(isinstance(artist, str) for artist in data["artists"]):
            return False, "All artists must be strings"
        return True, data["artists"]
    except Exception as e:
        return False, str(e)

def list_festivals():
    if not os.path.exists(DATA_DIR):
        return []
    items = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and not d.startswith('.')
    ]
    return sorted(items)

def get_artists_path(festival_dir):
    return os.path.join(DATA_DIR, festival_dir, "artists.json")

def select_festival(debug=False):
    while True:
        festivals = list_festivals()
        if not festivals:
            click.echo("No festivals found in the data directory.")
            sys.exit(1)
        click.echo("Select the festival for the playlist:")
        for idx, fest in enumerate(festivals, 1):
            click.echo(f"{idx}. {fest}")
        festival_idx = click.prompt("Enter festival number", type=int)
        if not (1 <= festival_idx <= len(festivals)):
            click.echo("Invalid selection. Please try again.")
            continue
        selected_festival = festivals[festival_idx - 1]
        artists_path = get_artists_path(selected_festival)
        is_valid, artists_or_err = validate_artists_json(artists_path)
        if is_valid:
            print_debug(f"Selected festival: {selected_festival}", debug)
            return selected_festival, artists_or_err
        else:
            click.echo(f"Error: {artists_or_err}")
            click.echo("The selected festival does not have a valid artists.json.")
            click.echo("Please choose another festival.\n")
            continue

def ask_for_inputs(playlist_name, code, sp_oauth, debug=False):
    if not playlist_name:
        playlist_name = click.prompt("Enter playlist name", type=str)
    # If code not provided, print auth URL and ask for code
    if not code:
        auth_url = sp_oauth.get_authorize_url()
        click.echo("\nTo authorize the app, open the following URL in your browser, log in, and then paste the code parameter you receive here:")
        click.echo(auth_url)
        code = click.prompt("\nPaste the 'code' parameter here", type=str)
    print_debug(f"Playlist name: {playlist_name}", debug)
    print_debug(f"Spotify code: {code}", debug)
    return playlist_name, code

def create_spotify_client(code, sp_oauth, debug=False):
    if not (SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI):
        click.echo("Spotify credentials are not set in the .env file.")
        sys.exit(1)
    token_info = sp_oauth.get_access_token(code=code, as_dict=False)
    print_debug(f"Token info: {token_info}", debug)
    return Spotify(auth=token_info)

def fetch_artist_popularity(sp, artists, debug=False):
    artist_popularity = {}
    click.echo("\n📊 Getting artist popularity...")
    for name in tqdm(artists, desc="Getting popularity"):
        res = sp.search(q=f"artist:{name}", type="artist", limit=1)
        items = res["artists"]["items"]
        artist_popularity[name] = items[0]["popularity"] if items else None
        time.sleep(0.2)
        if debug:
            print_debug(f"Artist '{name}' popularity: {artist_popularity[name]}", debug)
    return artist_popularity

def fetch_artist_top_tracks(sp, artist_name, debug=False):
    res = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    items = res["artists"]["items"]
    if not items:
        if debug:
            print_debug(f"Artist '{artist_name}' not found in Spotify", debug)
        return []
    found_artist = items[0]["name"]
    if normalize(found_artist) != normalize(artist_name):
        if debug:
            print_debug(f"Artist '{artist_name}' not matched strictly. Found '{found_artist}'", debug)
        return []
    artist_id = items[0]["id"]
    tracks = sp.artist_top_tracks(artist_id)["tracks"]
    top_track_ids = [track["id"] for track in tracks]
    if debug:
        print_debug(f"Artist '{artist_name}' top tracks: {top_track_ids}", debug)
    return top_track_ids

def get_percentile_thresholds(popularities):
    p20, p40, p60, p80 = np.percentile(popularities, [20, 40, 60, 80])
    return p20, p40, p60, p80

def songs_count(p, p20, p40, p60, p80):
    if p is None:
        p = 0
    if p < p20:
        return 1
    elif p < p40:
        return 2
    elif p < p60:
        return 3
    elif p < p80:
        return 4
    else:
        return 5
    
def normalize(text: str) -> str: 
    """Normalize string: remove accents and lowercase"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def select_tracks_by_percentile_logic(sp, artists, debug=False):
    """
    Assigns tracks to the first artist in the track's artists list that also appears in the user's artist list (artists.json).
    Each track is only assigned once (to the owner artist).
    The comparison is accent-insensitive and case-insensitive.
    """
    # Map normalized artist names to their order of preference (index) for quick lookup
    artist_preference = {normalize(name): i for i, name in enumerate(artists)}
    # Also keep a mapping from normalized to original for debug clarity
    artist_name_map = {normalize(name): name for name in artists}

    artist_popularity = fetch_artist_popularity(sp, artists, debug=debug)
    popularity_values = [
        artist_popularity[a] if artist_popularity[a] is not None else 0 for a in artists
    ]
    p20, p40, p60, p80 = get_percentile_thresholds(popularity_values)
    to_extract = {
        a: songs_count(artist_popularity[a], p20, p40, p60, p80) for a in artists
    }
    seen_tracks = set()
    all_tracks = []
    artist_to_count = {a: 0 for a in artists}
    debug_tracks_zero = {}
    click.echo("\n🎵 Fetching top tracks for each artist...")
    for artist in tqdm(artists, desc="Fetching tracks"):
        top_tracks = fetch_artist_top_tracks(sp, artist, debug=debug)
        rejected_tracks = []
        for track_id in top_tracks:
            reason = None
            if track_id in seen_tracks:
                reason = "already assigned to previous artist"
                rejected_tracks.append((track_id, reason, None, None))
                continue  # already assigned
            track_info = sp.track(track_id)
            # Find the first artist in track's authors that is in our artists list (accent/case-insensitive)
            owner = None
            authors = [auth["name"] for auth in track_info["artists"]]
            for auth_name in authors:
                norm_auth_name = normalize(auth_name)
                if norm_auth_name in artist_preference:
                    owner = artist_name_map[norm_auth_name]
                    break
            if owner != artist:
                reason = f"owner is {owner if owner else 'None'}"
                rejected_tracks.append((track_id, reason, authors, owner))
                continue  # not the owner for this song
            if artist_to_count[artist] < to_extract[artist]:
                all_tracks.append(track_id)
                seen_tracks.add(track_id)
                artist_to_count[artist] += 1
            else:
                reason = "limit reached for artist"
                rejected_tracks.append((track_id, reason, authors, owner))
            if artist_to_count[artist] >= to_extract[artist]:
                break  # got enough songs for this artist
        if artist_to_count[artist] == 0 and debug:
            debug_tracks_zero[artist] = {
                'top_tracks': top_tracks,
                'rejected': rejected_tracks
            }
        if debug:
            print_debug(f"{artist} | Popularity: {artist_popularity[artist]}, Tracks taken: {artist_to_count[artist]}", debug)
        time.sleep(0.2)
    # Print debug for artists with 0 tracks
    if debug and debug_tracks_zero:
        click.echo("\n[DEBUG] Artists with no assigned tracks:")
        for artist, info in debug_tracks_zero.items():
            click.echo(f"\n  - {artist} (requested: {to_extract[artist]})")
            for track_id, reason, authors, owner in info['rejected']:
                click.echo(f"    Track: {track_id}")
                if authors:
                    click.echo(f"      Authors: {authors}")
                if owner is not None:
                    click.echo(f"      Detected owner: {owner}")
                click.echo(f"      Rejected: {reason}")
    return all_tracks, artist_popularity, to_extract



def add_tracks_to_playlist_with_progress(sp, playlist_id, tracks):
    chunk_size = 100
    click.echo("Uploading tracks to Spotify...")
    for i in tqdm(range(0, len(tracks), chunk_size), desc="Uploading"):
        sp.playlist_add_items(playlist_id, tracks[i:i+chunk_size])

def print_summary(playlist_name, festival, num_artists, num_tracks):
    click.echo("\nSummary:")
    click.echo(f"Playlist name: {playlist_name}")
    click.echo(f"Festival: {festival}")
    click.echo(f"Number of artists: {num_artists}")
    click.echo(f"Number of tracks to add: {num_tracks}\n")

@click.command()
@click.option('--playlist-name', default=None, help="Name for the new Spotify playlist.")
@click.option('--code', default=None, help="Spotify authorization code.")
@click.option('--festival', default=None, help="Festival directory name (will prompt if not set).")
@click.option('--debug', is_flag=True, help="Enable debug mode with extra output.")
def main(playlist_name, code, festival, debug):
    print_debug("Starting HitTheFest CLI", debug)
    # Festival selection
    if not festival:
        selected_festival, artists = select_festival(debug=debug)
    else:
        artists_path = get_artists_path(festival)
        is_valid, artists_or_err = validate_artists_json(artists_path)
        if not is_valid:
            click.echo(f"Error: {artists_or_err}")
            sys.exit(1)
        selected_festival, artists = festival, artists_or_err

    # Setup SpotifyOAuth object (needed for both auth url and token exchange)
    if not (SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET and SPOTIPY_REDIRECT_URI):
        click.echo("Spotify credentials are not set in the .env file.")
        sys.exit(1)
    sp_oauth = SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="playlist-modify-public playlist-modify-private",
    )

    # Playlist name and Spotify code (show auth_url if needed)
    playlist_name, code = ask_for_inputs(playlist_name, code, sp_oauth, debug=debug)

    # Connect to Spotify
    sp = create_spotify_client(code, sp_oauth, debug=debug)
    user_id = sp.current_user()["id"]

    # Select tracks using notebook logic (percentile, no duplicates)
    tracks_to_add, artist_popularity, to_extract = select_tracks_by_percentile_logic(sp, artists, debug=debug)

    print_summary(playlist_name, selected_festival, len(artists), len(tracks_to_add))
    confirm = click.prompt("Do you want to create and upload the playlist? [Y/n]", default="Y")
    if confirm.lower() not in ["y", "yes", ""]:
        click.echo("Aborted.")
        sys.exit(0)

    # Create playlist and add tracks (with tqdm for progress)
    click.echo("Creating playlist on Spotify...")
    new_playlist = sp.user_playlist_create(user_id, playlist_name)
    add_tracks_to_playlist_with_progress(sp, new_playlist["id"], tracks_to_add)
    click.echo(f"Playlist '{playlist_name}' created with {len(tracks_to_add)} tracks!")

    if debug:
        click.echo("\nExtra debug output:")
        for artist in artists:
            click.echo(f"{artist}: Popularity {artist_popularity[artist]}, Tracks to extract: {to_extract[artist]}")

if __name__ == "__main__":
    main()
