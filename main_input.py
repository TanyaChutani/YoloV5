import argparse
from input_pipeline import InputPipeline

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-ww",
                      '--width',
                      type=int,
                      metavar='',
                      default = 1024)

  parser.add_argument('-hh',
                      '--height',
                      type=int,
                      metavar='',
                      default = 1024)

  parser.add_argument('-p',
                      '--file_path',
                      type=str,
                      metavar='',
                      default = 'trainval/annotations/bbox-annotations.json')
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  input = InputPipeline(input_file=args.file_path,
                        width=args.width,
                        height=args.height)
