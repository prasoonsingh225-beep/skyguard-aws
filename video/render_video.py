from pathlib import Path

# Optional helper: print the exact commands used by the video package.
root = Path(__file__).parent
print('Run: bash', root / 'build_video.sh')
print('Output: ', root / 'aero_guard_sih_26073.mp4')
