from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm


def _iter_response_bytes(response, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    if type(response).__name__ == "HTTPResponse":
        while True:
            data = response.read(chunk_size)
            if not data:
                break
            yield data
        return

    for data in response.iter_content(chunk_size):
        if not data:
            break
        yield data


def _save_response(response, outfile: Path, quiet: bool = True) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    headers = response.headers or {}
    size_header = headers.get("Content-Length")
    expected_size = int(size_header) if size_header and size_header.isdigit() else None
    last_modified = headers.get("Last-Modified")

    if not quiet:
        print(f"Downloading {outfile.name} -> {outfile.parent}")

    written = 0
    with tqdm(
        total=expected_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=quiet,
    ) as pbar:
        with outfile.open("wb") as f:
            for chunk in _iter_response_bytes(response):
                f.write(chunk)
                written += len(chunk)
                pbar.update(len(chunk))

    if expected_size is not None and written != expected_size:
        raise ValueError(
            f"Downloaded file size ({written}) does not match expected size ({expected_size}) for {outfile}"
        )

    if last_modified:
        try:
            remote_date = datetime.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")
            remote_ts = remote_date.timestamp()
            os.utime(outfile, times=(remote_ts, remote_ts))
        except ValueError:
            pass

    if not quiet:
        print(f"Saved {outfile}")
    return outfile


class RobustKaggleApi(KaggleApi):
    def download_file(self, response, outfile, http_client, quiet=True, **kwargs):  # type: ignore[override]
        del http_client, kwargs
        _save_response(response, Path(outfile), quiet=quiet)


def build_api() -> RobustKaggleApi:
    api = RobustKaggleApi()
    api.authenticate()
    return api


def _authenticated_client():
    api = build_api()
    kaggle = api.build_kaggle_client()
    kaggle.__enter__()
    client = kaggle.competitions.competition_api_client._client
    client._init_session()
    return api, kaggle, client


def _download_via_endpoint(
    request_name: str,
    payload: dict[str, int],
    outfile: Path,
    quiet: bool,
) -> Path:
    api, kaggle, client = _authenticated_client()
    try:
        url = client._get_request_url("competitions.CompetitionApiService", request_name)
        headers = dict(client._session.headers)
        headers["Accept-Encoding"] = "identity"
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            auth=client._session.auth,
            stream=True,
            timeout=300,
        )
        response.raise_for_status()
        return _save_response(response, outfile, quiet=quiet)
    finally:
        kaggle.__exit__(None, None, None)
        del api


def list_episodes(submission_id: int, csv_display: bool = False) -> None:
    api = build_api()
    episodes = api.competition_list_episodes(submission_id)
    if csv_display:
        print("id,createTime,endTime,state,type")
        for episode in episodes:
            print(
                f"{episode.id},{episode.create_time},{episode.end_time},{episode.state},{episode.type}"
            )
        return

    for episode in episodes:
        print(
            f"{episode.id}  {episode.create_time}  {episode.end_time}  {episode.state}  {episode.type}"
        )


def download_replay(episode_id: int, path: str, quiet: bool = False) -> None:
    outfile = Path(path) / f"episode-{episode_id}-replay.json"
    _download_via_endpoint("GetEpisodeReplay", {"episodeId": episode_id}, outfile, quiet=quiet)


def download_logs(episode_id: int, agent_index: int, path: str, quiet: bool = False) -> None:
    outfile = Path(path) / f"episode-{episode_id}-agent-{agent_index}-logs.json"
    _download_via_endpoint(
        "GetEpisodeAgentLogs",
        {"episodeId": episode_id, "agentIndex": agent_index},
        outfile,
        quiet=quiet,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust Kaggle competition episode tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    episodes_parser = subparsers.add_parser("episodes", help="List episodes for a submission")
    episodes_parser.add_argument("submission_id", type=int)
    episodes_parser.add_argument("-v", "--csv", action="store_true", help="Print CSV output")

    replay_parser = subparsers.add_parser("replay", help="Download a replay for an episode")
    replay_parser.add_argument("episode_id", type=int)
    replay_parser.add_argument("-p", "--path", default=".", help="Output directory")
    replay_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    logs_parser = subparsers.add_parser("logs", help="Download logs for an agent in an episode")
    logs_parser.add_argument("episode_id", type=int)
    logs_parser.add_argument("agent_index", type=int)
    logs_parser.add_argument("-p", "--path", default=".", help="Output directory")
    logs_parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()
    if args.command == "episodes":
        list_episodes(args.submission_id, csv_display=args.csv)
    elif args.command == "replay":
        download_replay(args.episode_id, path=args.path, quiet=args.quiet)
    elif args.command == "logs":
        download_logs(args.episode_id, args.agent_index, path=args.path, quiet=args.quiet)


if __name__ == "__main__":
    main()
