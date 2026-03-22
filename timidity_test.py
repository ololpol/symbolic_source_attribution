from timidity import Parser, play_notes
import numpy as np

ps = Parser("folkrnn/data/midi/sessiontune45843.mid")

play_notes(*ps.parse(), np.sin)