from faster_whisper import WhisperModel

import argparse

def parseargs():
  parser = argparse.ArgumentParser(description='')

  parser.add_argument("-i",
    "--input",
    type=str,
    help="input file name")


  return parser.parse_args()


def main():
  args = parseargs()

  model_size = "large-v2"

  # Run on GPU with FP16
  # model = WhisperModel(model_size, device="cuda", compute_type="float16")

  # or run on GPU with INT8
  # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
  # or run on CPU with INT8
  model = WhisperModel(model_size, device="cpu", compute_type="int8")

  segments, info = model.transcribe(args.input, beam_size=5, vad_filter=True)

  print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

  for segment in segments:
      print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
      # print("%s" % (segment.text))

if __name__ == "__main__":
  main()