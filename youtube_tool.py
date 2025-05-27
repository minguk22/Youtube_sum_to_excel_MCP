from httpx import get
import streamlit as st

# OpenAI 패키지 추가
import openai
from openai import OpenAI

# 유튜브 영상을 다운로드하기 위한 pytube 패키지 추가
from pytubefix import YouTube
import pandas as pd
import openpyxl

# 유튜브 영상을 번역, 요약하기 위한 Langchain 패키지 추가
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


# 필요한 기본 패키지 추가
import re #re은 이 코드에서 정규 표현식(Regex)을 사용하기 위해서 추가.
import os # os는 파일을 다루기 위해서 추가.
import shutil # shutil은 파일을 이동하기 위해서 추가.
from dotenv import load_dotenv

from mcp.server.fastmcp.server import FastMCP

api_key = os.getenv("OPENAI_API_KEY")
mcp=FastMCP()

client = OpenAI(api_key=api_key)

def get_metadata(url):
    """
    유튜브 영상의 URL을 입력받아 해당 영상의 메타데이터를 반환하는 함수.
    """
    yt = YouTube(url)
    metadata = {
        "title": yt.title,
        "author": yt.author,
        "length": yt.length,
        "description": yt.description,
        "publish_date": yt.publish_date,
        "views": yt.views,
        "keywords": yt.keywords,
        "thumbnail_url": yt.thumbnail_url,
        "video_id": yt.video_id,
        "channel_id": yt.channel_id
    }
    return metadata

# 주소를 입력받으면 유튜브 동영상의 음성(mp3)을 추출하는 함수.
def get_audio(url):
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()# 오디오 스트림을 가져옴.
    audio_file = audio.download(output_path=".")# 다운로드한 파일의 경로를 지정.
    base, ext = os.path.splitext(audio_file) # 확장자 분리
    new_audio_file = base + '.mp3'# mp3로 변환
    shutil.move(audio_file, new_audio_file)# mp4를 mp3로 변환
    return new_audio_file# mp3 파일의 경로를 반환.

# 음성 파일 위치를 전달받으면 스크립트를 추출.
def get_transcribe(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
        # transcript가 str이 아닐 경우 대비
        if not transcript or not isinstance(transcript, str):
            return ""
        return transcript

# 영어 입력이 들어오면 한글로 번역 및 불렛포인트 요약을 수행.
def trans(text):
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 영한 번역가이자 요약가입니다. 들어오는 모든 입력을 한국어로 번역하고 불렛 포인트 요약을 사용하여 답변하시오. 반드시 불렛 포인트 요약이어야만 합니다."},
                    {"role": "user", "content": text}
                ]
            )
    return response.choices[0].message.content

def split(transcript):
    """유튜브 영상의 스크립트를 입력받아 텍스트를 분할하는 함수."""
    if not transcript:
        return []
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    text = splitter.create_documents([transcript])
    return text

def summarize(text):
    """유튜브 영상의 스크립트를 입력받아 요약하는 함수."""
    llm = ChatOpenAI(temperature=0,
        openai_api_key=api_key,
        max_tokens=4000,
        model_name="gpt-3.5-turbo",
        request_timeout=120
    )
    map_prompt = PromptTemplate(
    template="""
    아래 텍스트는 유튜브 영상의 일부분입니다.
    1. 이 부분의 주요 내용을 타임라인 순서대로 간단히 정리하세요.
    2. 이 부분에서 중요한 키포인트(핵심 메시지, 인사이트, 사건 등)를 bullet point로 정리하세요.
    3. 요약은 한국어로 작성하세요.
    텍스트:
    ```{text}```
    ---
    [출력 예시]
    타임라인 요약:
    - ...
    - ...
    키포인트:
    - ...
    - ...
    """,
    input_variables=["text"]
    )
    # combine_prompt: 모든 chunk 요약을 종합해 전체 타임라인/키포인트 요약
    combine_prompt = PromptTemplate(
        template="""
    아래는 유튜브 영상의 여러 부분 요약과 키포인트입니다.
    1. 전체 타임라인을 순서대로 요약하세요.
    2. 전체 영상에서 가장 중요한 키포인트만 bullet point로 정리하세요.
    3. 요약은 한국어로 작성하세요.

    부분 요약들:
    ```{text}```
    ---
    [출력 예시]
    전체 타임라인 요약:
    - ...
    - ...
    전체 키포인트:
    - ...
    - ...
    """,
    input_variables=["text"]
    )

    chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False,
                            map_prompt=map_prompt, combine_prompt=combine_prompt)
    return chain.invoke(text).get("output_text")

@mcp.tool()
def youtube_summarize_excel(urls):
    """
    여러 유튜브 영상 URLS을 받아 각각 처리 후, 결과를 하나의 엑셀 파일에 저장합니다.
    """
    results = []
    for url in urls:
        try:
            yt = get_metadata(url)
            audio_file = get_audio(url)
            transcript = get_transcribe(audio_file)
            # 오디오 파일 삭제
            if os.path.exists(audio_file):
                 os.remove(audio_file)
            text = split(transcript)
            summary = summarize(text)
            # publish_date에서 timezone 제거
            publish_date = yt["publish_date"]
            if hasattr(publish_date, "tzinfo") and publish_date.tzinfo is not None:
                publish_date = publish_date.replace(tzinfo=None)
            results.append({
                "Title": yt["title"],
                "Length (sec)": yt["length"],
                "Description": yt["description"],
                "Publish Date": publish_date,
                "Keywords": ", ".join(yt["keywords"]),
                "Transcript": transcript,
                "Summary": summary
            })
        except Exception as e:
            print(f"Error processing {url}: {e}")
        
        df = pd.DataFrame(results)
        output_file = "youtube_summary_multi.xlsx"
        df.to_excel(output_file, index=False)
        print(f"{len(results)}개의 유튜브 영상 정보가 '{output_file}' 파일에 저장되었습니다.")


if __name__ == "__main__":
    urls = [

        # 필요시 추가
    ]
    result = youtube_summarize_excel(urls)
    print(result)
    mcp.run()



