import requests
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Path to save uploaded files and processed videos
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

# Load YOLOv8 models
model_train4 = YOLO(r"C:\Users\boazd\runs\detect\train4\weights\best.pt")
model_train5 = YOLO(r"C:\Users\boazd\runs\detect\train5\weights\best.pt")

# ======================================
# ROUTES FOR STATIC PAGES (HTML TEMPLATES)
# ======================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game-scores')
def game_scores():
    return render_template('game-scores.html')

@app.route('/game-stats')
def game_stats():
    return render_template('game-stats.html')

@app.route('/player-profiles')
def player_profiles():
    return render_template('player-profiles.html')

@app.route('/team-profiles')
def team_profiles():
    return render_template('team-profiles.html')

@app.route('/league-standings')
def league_standings():
    return render_template('league-standings.html')

@app.route('/upload')
def upload_video():
    return render_template('upload.html')

@app.route('/results')
def results():
    raw_video = request.args.get('raw_video')
    highlight_video = request.args.get('highlight_video')

    if not raw_video or not highlight_video:
        return "Error: Missing video files", 400

    return render_template('results.html', raw_video=raw_video, highlight_video=highlight_video)



# ======================================
# NBA Standings API Route
# ======================================
@app.route('/api/standings/<int:year>', methods=['GET'])
def get_standings(year):
    try:
        # Construct the season string in the format "2022-23"
        season = f"{year}-{str(year + 1)[-2:]}"
        
        # NBA API endpoint for standings
        url = "https://stats.nba.com/stats/leaguestandings"
        
        # API parameters
        params = {
            'LeagueID': '00',  # 00 for NBA
            'Season': season,
            'SeasonType': 'Regular Season'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/',
            'Connection': 'keep-alive'
        }

        # Send the request to the NBA API
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch standings data'}), 500

        # Parse JSON data
        data = response.json()
        
        # Extract the standings data
        standings = data['resultSets'][0]['rowSet']
        headers = data['resultSets'][0]['headers']
        
        # Format the standings in a usable structure
        standings_data = [dict(zip(headers, team)) for team in standings]

        # Return JSON to the frontend
        return jsonify(standings_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ======================================
# ROUTE FOR NBA LEAGUE LEADERS
# ======================================
@app.route('/api/leagueleaders', methods=['GET'])
def get_league_leaders():
    """Fetch league leaders from the NBA Stats API"""
    try:
        # Get the parameters from the request query string.
        season = request.args.get('Season')
        season_type = request.args.get('SeasonType')
        stat_category = request.args.get('StatCategory')
        per_mode = request.args.get('PerMode', 'Totals')  # Default to 'Totals' if not provided
        scope = request.args.get('Scope', 'S')            # Default to 'S' (season) if not provided

        # NBA Stats API endpoint for league leaders
        url = "https://stats.nba.com/stats/leagueleaders"

        # Required parameters for the NBA Stats API
        params = {
            'LeagueID': '00',                # NBA league
            'PerMode': per_mode,             # Stats mode: Totals, PerGame, Per48
            'Scope': scope,                  # Scope: RS (regular season), S (season), Rookies
            'Season': season,                # Season (e.g., 2019-20)
            'SeasonType': season_type,       # Regular Season, Playoffs, Pre Season, All Star
            'StatCategory': stat_category    # Stat category (e.g., PTS, REB, AST)
        }

        # Headers to handle CORS and ensure successful request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Accept': 'application/json, text/plain, */*',
            'Origin': 'https://www.nba.com',
            'Referer': 'https://www.nba.com/',
            'Connection': 'keep-alive'
        }

        # Make a request to the NBA Stats API and return the data
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch league leaders data'}), 500

        data = response.json()

        # Return the data as JSON to the frontend
        return jsonify(data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ======================================
# ROUTES FOR VIDEO PROCESSING
# ======================================

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        print("No video file provided")
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        print("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded video to the UPLOAD_FOLDER
        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)
        print(f"Video saved to {video_path}")

        # Process the video and generate the raw and highlight versions
        raw_video = os.path.join(PROCESSED_FOLDER, 'raw.mp4')
        highlight_video = os.path.join(PROCESSED_FOLDER, 'highlight_video.mp4')

        # Step 1: Analyze the video for all objects and save as raw.mp4
        if analyze_video_for_all(video_path, raw_video):
            # Step 2: Analyze the raw.mp4 for made-basket events and create highlights
            analyze_video_for_made_basket(raw_video, highlight_video)
            print(f"Processed video and highlight generated: {raw_video}, {highlight_video}")

        # Return the filenames as a JSON response
        return jsonify({
            "rawVideo": "raw.mp4",
            "highlightVideo": "highlight_video.mp4"
        })

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": str(e)}), 500


# Serve processed videos from the "processed" directory
@app.route('/videos/<filename>')
def serve_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)



# ======================================
# VIDEO PROCESSING FUNCTIONS
# ======================================

def analyze_video_for_all(input_video_path, output_video_path):
    """Analyze the video for all objects (using train4)"""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Predict objects in the frame using model_train4
        results = model_train4(frame, save=False)

        # Draw bounding boxes on the frame
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = box.conf.item()
                xyxy = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
                label = model_train4.names[cls_id]  # Class label

                # Draw bounding box
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to {output_video_path}")
    return True

def analyze_video_for_made_basket(input_video_path, output_highlight_path):
    """Analyze the video to detect 'made-basket' events using the train5 model."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    highlight_out = cv2.VideoWriter(output_highlight_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_counter = 0
    made_basket_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict objects in the frame using model_train5
        results = model_train5(frame, save=False)

        # Detect 'made-basket' events
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = box.conf.item()
                label = model_train5.names[cls_id]  # Class label

                if label == 'made-basket' and conf > 0.5:
                    made_basket_frames.append(frame_counter)

        frame_counter += 1

    cap.release()

    # Highlight generation (slicing based on frames)
    if made_basket_frames:
        time_windows = create_time_stamp_windows(made_basket_frames, 7, fps, cooldown_seconds=2)
        slice_video(input_video_path, highlight_out, time_windows)
    else:
        print("No made-basket events detected.")

    highlight_out.release()

def create_time_stamp_windows(made_basket_frames, seconds_before, fps, cooldown_seconds=2):
    start_frames = np.clip(np.array(made_basket_frames) - (seconds_before * fps), 0, None)
    final_frames = []
    last_end_frame = -float('inf')

    for start_frame in start_frames:
        if start_frame >= last_end_frame + (cooldown_seconds * fps):
            end_frame = start_frame + (seconds_before * fps)
            final_frames.append((start_frame, end_frame))
            last_end_frame = end_frame
    
    return final_frames

def slice_video(input_video_path, highlight_out, time_windows):
    cap = cv2.VideoCapture(input_video_path)

    frame_counter = 0
    highlight_counter = 0
    start_frames, end_frames = zip(*time_windows)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if highlight_counter >= len(end_frames):
            break

        if start_frames[highlight_counter] <= frame_counter <= end_frames[highlight_counter]:
            highlight_out.write(frame)

        if frame_counter > end_frames[highlight_counter]:
            highlight_counter += 1

        frame_counter += 1

    cap.release()
    print("Highlight video generated.")


# ======================================
# RUN THE APPLICATION
# ======================================
if __name__ == "__main__":
    app.run(debug=True)
