import ffmpeg
import subprocess

stream = ffmpeg.input('F:\python\MyProject\deepFake\data\mrbean_superman.mp4')
stream = ffmpeg.output(stream, 'output.mp4')
ffmpeg.run(stream)

# subprocess.run(stream)
