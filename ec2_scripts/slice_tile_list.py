import polars as pl
from argparse import ArgumentParser
from pathlib import Path

def load_and_slice(file, start, length):
    df = pl.scan_csv(file)
    slice = df.slice(start, length)
    urls = slice.select('url').collect().to_series()
    return urls.to_list()

def write_urls(lst, path = Path(__file__).parent / 'production_infer.txt'):
    out_text = "\n".join(lst)
    with open(path, 'w') as f:
        f.write(out_text)
    return path

def main():
    parser = ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("start", type = int)
    parser.add_argument("length", type = int)
    args = parser.parse_args()
    urls = load_and_slice(args.file, args.start, args.length)
    path = write_urls(urls)
    print(path)

if __name__ == '__main__':
    main()