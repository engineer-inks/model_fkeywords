class Colorize:
    styles = {
        'default': 0,  # Turn off all attributes
        'bold': 1,  # Set bold mode
        'unable': 2,  # Set unable mode
        'italic': 3,  # Set italic mode
        'underline': 4,  # Set underline mode
        'blink': 5,  # Set blink mode
        'swap': 7,  # Exchange foreground and background colors
        # Hide text (foreground color would be the same as background)
        'hide': 8,
        'strykethrough': 9,  # Exchange foreground and background colors
    }

    colors = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
        'default': 39,
        'on_black': 40,
        'on_red': 41,
        'on_green': 42,
        'on_yellow': 43,
        'on_blue': 44,
        'on_magenta': 45,
        'on_cyan': 46,
        'on_white': 47,
    }

    @classmethod
    def get_color(cls, text: str, color='default', style='default'):
        """
        Returns the text with the given style

        :param text: Text that will be written
        :type text: str
        :param color: Color of text, defaults to 'default'
        :type color: str, optional
        :param style: Style of text, defaults to 'default'
        :type style: str, optional
        :return: Text with styles and colors applied
        :rtype: str
        """
        return f'\x1b[{cls.styles[style]};{cls.colors[color]}m{text}\x1b[0m'
