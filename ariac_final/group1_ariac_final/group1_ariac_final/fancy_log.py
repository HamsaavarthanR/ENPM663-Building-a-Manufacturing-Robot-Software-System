class FancyLog:
    """
    A utility class for fancy terminal logging with colors and text styles.
    Similar to the color formatting in the C++ version of the code.
    """

    # Basic color definitions
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"

    # Bright versions
    BRIGHT_BLACK = "\033[90m"  # Usually appears as dark gray
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Text styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALIC = "\033[3m"  # Not supported by all terminals

    # Custom colors matching the C++ implementation
    ORANGE = "\033[38;2;255;165;0m"
    PURPLE = "\033[38;2;128;0;128m"
    LIGHT_BLUE = "\033[38;2;173;216;230m"
    CHARM_PINK = "\033[38;2;235;143;166m"

    @staticmethod
    def rgb_color(r, g, b):
        """
        Create a custom color from RGB values (true color).

        Args:
            r (int): Red component (0-255)
            g (int): Green component (0-255)
            b (int): Blue component (0-255)

        Returns:
            str: ANSI escape code for the specified RGB color
        """
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def color_256(code):
        """
        Create a custom color from 256-color palette.

        Args:
            code (int): Color code from the 256-color palette (0-255)

        Returns:
            str: ANSI escape code for the specified color
        """
        return f"\033[38;5;{code}m"

    @staticmethod
    def format_text(text, *formats):
        """
        Apply one or more format codes to text and reset afterwards.

        Args:
            text (str): The text to format
            *formats: Variable number of format codes to apply

        Returns:
            str: The formatted text with reset code appended
        """
        return "".join(formats) + text + FancyLog.RESET

    @staticmethod
    def info(logger, text):
        """
        Log text with specified formats at INFO level.

        Args:
            logger: The ROS logger to use
            text (str): The text to log
        """
        logger.info(FancyLog.BOLD + FancyLog.CHARM_PINK + text + FancyLog.RESET)

    # @staticmethod
    # def warn(logger, text, *formats):
    #     """
    #     Log text with specified formats at WARN level.

    #     Args:
    #         logger: The ROS logger to use
    #         text (str): The text to log
    #         *formats: Variable number of format codes to apply
    #     """
    #     if formats:
    #         logger.warn("".join(formats) + text + FancyLog.RESET)
    #     else:
    #         logger.warn(text)
            
    @staticmethod
    def warn(logger, text):
        """
        Log text with specified formats at WARN level.

        Args:
            logger: The ROS logger to use
            text (str): The text to log
        """
        logger.info(FancyLog.BOLD + FancyLog.BRIGHT_YELLOW + text + FancyLog.RESET)

    @staticmethod
    def error(logger, text):
        """
        Log text with specified formats at ERROR level.

        Args:
            logger: The ROS logger to use
            text (str): The text to log
        """
        logger.info(FancyLog.BOLD + FancyLog.RED + text + FancyLog.RESET)
        
    @staticmethod
    def pscene(logger, text):
        """
        Log text with specified formats for messages related to the planning scene

        Args:
            logger: The ROS logger to use
            text (str): The text to log
        """
        logger.info(FancyLog.BOLD + FancyLog.BRIGHT_YELLOW + text + FancyLog.RESET)
        
        

    @staticmethod
    def debug(logger, text, *formats):
        """
        Log text with specified formats at DEBUG level.

        Args:
            logger: The ROS logger to use
            text (str): The text to log
            *formats: Variable number of format codes to apply
        """
        if formats:
            logger.debug("".join(formats) + text + FancyLog.RESET)
        else:
            logger.debug(text)
