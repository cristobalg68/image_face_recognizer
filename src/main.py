import sys
import os
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # apply theme
    qss_path = os.path.join(os.path.dirname(__file__), "styles", "theme.qss")
    try:
        with open(qss_path, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except Exception:
        pass

    window = MainWindow()
    window.show()
    sys.exit(app.exec())