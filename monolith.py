import os
import json
import tempfile
import time
import asyncio
import nest_asyncio
import cv2
import requests
import importlib
import yaml
import numpy as np
from tqdm.auto import tqdm
from openai import AsyncOpenAI
import matplotlib.pyplot as plt 

nest_asyncio.apply()
from IPython.display import Image
from IPython.core.display import HTML, Markdown

from utils import ( split_video, generate_subtitles, get_relevant_frames, \
                    get_sectionwise_quality_frames, upload_image_to_imgbb )
from hparams import *
from constants import *
import logging


logging.getLogger().setLevel(logging.WARNING)


def load_secrets():
    with open('secrets.yaml', 'r') as file:
        secrets = yaml.load(file, Loader=yaml.FullLoader)

    for key, value in secrets.items():
        os.environ[key] = value


def split_video_into_chunks(metadata):
    split_video(VIDEO_PATH, BASE_DIR, duration=MAX_DURATION_SEC)
    logging.info(f"Video Saved into chunks: {BASE_DIR} with max duration as: {MAX_DURATION_SEC}")

def save_subtitles(metadata):
    subtitles, sectionwise_subtitles = asyncio.run(generate_subtitles(metadata['openai_client'], BASE_DIR, duration=MAX_DURATION_SEC))
    with open(SRT_PATH, "w", encoding="utf-8") as file:
        file.write(subtitles)
    logging.info(f"Subtitles Saved in: {SRT_PATH}")

def shortlist_frames():
    shortlisted_frames = get_relevant_frames(VIDEO_PATH, duration=MAX_DURATION_SEC, seconds_between_frames=SECONDS_BETWEEN_FRAMES, similarity_threshold=SSIM_THRESHOLD)
    logging.info(f"Shortlisted: {sum([len(v) for k,v in shortlisted_frames.items()])} after SSIM")

def sectionwise_shortlist_frames():
    sectionwise_shortlisted_frames = get_sectionwise_quality_frames(shortlisted_frames)
    logging.info(f"Total selected frames section wise: {sum([len(v) for v in sectionwise_shortlisted_frames.values()])}")

def save_frames():
    if not os.path.exists(os.path.join(BASE_DIR,FRAMES_DIR)):  # This checks if the save directory exists, creates if it don't
        os.makedir(os.path.join(BASE_DIR,FRAMES_DIR))
    SAVE_DIR = os.path.join(BASE_DIR,FRAMES_DIR)
    for section, section_frames in sectionwise_shortlisted_frames.items():
        NSHOW_IMAGES = min(20, len(section_frames))
        NCOL = 10
        NROW = NSHOW_IMAGES // NCOL + 1
        if NSHOW_IMAGES == 0:
            continue
        
        for i, frame in tqdm(enumerate(section_frames[:NSHOW_IMAGES])):         
            save_path = os.path.join(SAVE_DIR, f"section_{section}_frame_{i}.png")
            cv2.imwrite(save_path, frame['frame']) 

def main():
    load_secrets()
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    pipeline = [
        split_video_into_chunks,
        save_subtitles,
        shortlist_frames,
        sectionwise_shortlist_frames,
        save_frames
    ]

    metadata = {"openai_client":openai_client}
    for function in pipeline:
        response = function(metadata)

if __name__ == "__main__":
    main()