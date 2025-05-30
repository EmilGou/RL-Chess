# python-chess-UI/src/theme.py
from color import Color # Your Color class that takes (light_value, dark_value)

class Theme:
    def __init__(self, 
                 light_bg_value,    # e.g., (235, 235, 208)
                 dark_bg_value,     # e.g., (119, 148, 85)
                 light_trace_value, 
                 dark_trace_value,
                 light_moves_value, 
                 dark_moves_value,
                 piece_style_folder: str, # <<<--- Make sure this is accepted
                 name: str):              # <<<--- Make sure this is accepted
        """
        Initializes a Theme object.

        Args:
            light_bg_value: RGB tuple for light background squares.
            dark_bg_value: RGB tuple for dark background squares.
            light_trace_value: RGB tuple for light trace squares (last move).
            dark_trace_value: RGB tuple for dark trace squares (last move).
            light_moves_value: RGB tuple for light moves squares (valid moves).
            dark_moves_value: RGB tuple for dark moves squares (captures).
            piece_style_folder: The name of the subfolder within assets/images/ 
                                containing the piece images for this theme (e.g., 'imgs-80px').
            name: The name of the theme (e.g., 'green').
        """
        
        # Group colors using your Color class
        self.bg = Color(light_bg_value, dark_bg_value)
        self.trace = Color(light_trace_value, dark_trace_value)
        self.moves = Color(light_moves_value, dark_moves_value)
        
        # Store the added attributes
        self.piece_style_folder = piece_style_folder 
        self.name = name                           
