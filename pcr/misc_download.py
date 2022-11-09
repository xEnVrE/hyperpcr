import requests
from pathlib import Path
from rich.progress import track


def download(url, dir, name):
    dir = Path(dir)
    file_path = dir / name
    full_url = f'{url}/{file_path.as_posix()}'

    if not dir.exists():
        dir.mkdir()

    if not file_path.exists():

        r = requests.get(full_url, stream=False)
        l = int(r.headers['Content-length']) / 1024

        with file_path.open('wb') as f:
            for chunk in track(r.iter_content(chunk_size=1024), total=l,
                               description=f'Downloading {dir.as_posix()[:-1]}...'):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    return file_path.as_posix()


def download_checkpoint(name):
    url = f'https://github.com/andrearosasco/hyperpcr/raw/main'
    dir = 'checkpoints'

    return download(url, dir, name)


def download_asset(name):
    url = f'https://github.com/andrearosasco/hyperpcr/raw/main'
    dir = 'assets'

    return download(url, dir, name)
