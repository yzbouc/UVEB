import os
import cv2
import multiprocessing
import time

def extract_frames(video_file, output_dir):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()

def process_video(video_file):
    video_name = os.path.splitext(video_file)[0].rsplit('_', 1)[0]
    file_name = os.path.splitext(os.path.basename(video_name))[0]
    print("file_name",file_name)
    output_dir = os.path.join("/data2/Our/challenge/gt/", file_name)
    os.makedirs(output_dir, exist_ok=True)
    extract_frames(video_file, output_dir)

if __name__ == "__main__":
    start = time.time()
    input_folder = "/data2/Our_data/our_dataset/challenge/gt/"  
    video_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".mp4")]
    pool = multiprocessing.Pool(processes=12)
    pool.map(process_video, video_files)
    pool.close()
    pool.join()
    end = time.time()
    print('time: ',end - start)
