#!/usr/bin/env python

import os
import sys
import pathlib
import platform
import threading
import traceback
import logging
from packaging import version

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QCheckBox, QComboBox,
    QSlider, QTextEdit, QFileDialog, QMessageBox, QScrollArea,
    QSplashScreen, QGroupBox, QProgressBar, QStatusBar, QFrame, QSpinBox
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)
from PyQt6.QtGui import (
    QPixmap, QIcon
)

import downloader
from version import __version__
import getortho

log = logging.getLogger(__name__)

CUR_PATH = os.path.dirname(os.path.realpath(__file__))


class SceneryDownloadWorker(QThread):
    """Worker thread for downloading scenery"""
    progress = pyqtSignal(str, dict)  # region_id, progress_data
    finished = pyqtSignal(str, bool)  # region_id, success
    error = pyqtSignal(str, str)  # region_id, error_message

    def __init__(self, dl_manager, region_id, download_dir):
        super().__init__()
        self.dl_manager = dl_manager
        self.region_id = region_id
        self.download_dir = download_dir

    def run(self):
        try:
            self.dl_manager.download_dir = self.download_dir
            region = self.dl_manager.regions.get(self.region_id)

            def progress_callback(progress_data):
                self.progress.emit(self.region_id, progress_data)

            success = region.install_release(
                progress_callback=progress_callback
            )
            self.finished.emit(self.region_id, success)

        except Exception as err:
            tb = traceback.format_exc()
            self.error.emit(self.region_id, str(err))
            log.error(tb)


class StyledButton(QPushButton):
    """Custom styled button with hover effects"""
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setStyleSheet(self._get_style())

    def _get_style(self):
        if self.primary:
            return """
                QPushButton {
                    background-color: #6eb6ff;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #FF8555;
                }
                QPushButton:pressed {
                    background-color: #E55B25;
                }
                QPushButton:disabled {
                    background-color: #666;
                    color: #999;
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #3A3A3A;
                    color: white;
                    border: 1px solid #555;
                    padding: 6px 12px;
                    border-radius: 4px;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #4A4A4A;
                    border-color: #6eb6ff;
                }
                QPushButton:pressed {
                    background-color: #2A2A2A;
                }
                QPushButton:disabled {
                    background-color: #2A2A2A;
                    color: #666;
                    border-color: #333;
                }
            """


class ModernSlider(QSlider):
    """Custom styled slider"""
    def __init__(self, orientation=Qt.Orientation.Horizontal):
        super().__init__(orientation)
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #3A3A3A;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #6eb6ff;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #FF8555;
            }
            QSlider::sub-page:horizontal {
                background: #6eb6ff;
                border-radius: 3px;
            }
        """)


class ModernSpinBox(QSpinBox):
    """Custom styled spinbox"""
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QSpinBox {
                background-color: #3A3A3A;
                border: 1px solid #555;
                padding: 6px;
                border-radius: 4px;
                color: white;
                min-width: 80px;
            }
            QSpinBox:focus {
                border-color: #6eb6ff;
            }
            QSpinBox::up-button {
                background-color: #6eb6ff;
                border: none;
                border-radius: 2px;
                width: 16px;
            }
            QSpinBox::up-button:hover {
                background-color: #FF8555;
            }
            QSpinBox::up-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 4px solid white;
                width: 0px;
                height: 0px;
            }
            QSpinBox::down-button {
                background-color: #6eb6ff;
                border: none;
                border-radius: 2px;
                width: 16px;
            }
            QSpinBox::down-button:hover {
                background-color: #FF8555;
            }
            QSpinBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid white;
                width: 0px;
                height: 0px;
            }
        """)


class ConfigUI(QMainWindow):
    """Main configuration UI window using PyQt6"""

    status_update = pyqtSignal(str)
    log_update = pyqtSignal(str)
    show_error = pyqtSignal(str)

    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ready = threading.Event()
        self.ready.clear()

        self.dl = downloader.OrthoManager(
            self.cfg.paths.scenery_path,
            self.cfg.paths.download_dir,
            noclean=self.cfg.scenery.noclean
        )

        self.running = False
        self.warnings = []
        self.errors = []
        self.show_errs = []

        # Download management
        self.download_workers = {}
        self.download_progress = {}

        # Setup UI
        self.init_ui()
        self.show_splash()

        # Connect signals
        self.status_update.connect(self.update_status_bar)
        self.log_update.connect(self.append_log)
        self.show_error.connect(self.display_error)

        # Start log update timer
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_logs)
        self.log_timer.start(1000)

        self.ready.set()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f'AutoOrtho ver {__version__}')
        self.setGeometry(100, 100, 900, 700)

        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QWidget {
                background-color: #1E1E1E;
                color: #E0E0E0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QTabWidget::pane {
                border: 1px solid #3A3A3A;
                background-color: #2A2A2A;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #2A2A2A;
                color: #999;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3A3A3A;
                color: #6eb6ff;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background-color: #3A3A3A;
                color: #E0E0E0;
            }
            QLineEdit {
                background-color: #3A3A3A;
                border: 1px solid #555;
                padding: 6px;
                border-radius: 4px;
                color: white;
            }
            QLineEdit:focus {
                border-color: #6eb6ff;
            }
            QTextEdit {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                padding: 4px;
                color: #E0E0E0;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 1px solid #555;
                background-color: #3A3A3A;
            }
            QCheckBox::indicator:checked {
                background-color: #6eb6ff;
                border-color: #6eb6ff;
            }
            QComboBox {
                background-color: #3A3A3A;
                border: 1px solid #555;
                padding: 6px;
                border-radius: 4px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #6eb6ff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #999;
                margin-right: 5px;
            }
            QGroupBox {
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                color: #6eb6ff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #E0E0E0;
            }
            QStatusBar {
                background-color: #2A2A2A;
                border-top: 1px solid #3A3A3A;
                color: #999;
            }
            QScrollBar:vertical {
                background-color: #2A2A2A;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
        """)

        # Set icon
        if platform.system() == 'Windows':
            icon_path = os.path.join(CUR_PATH, 'imgs', 'ao-icon.ico')
        else:
            icon_path = os.path.join(CUR_PATH, 'imgs', 'ao-icon.png')
        self.setWindowIcon(QIcon(icon_path))

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 10)
        central_widget.setLayout(main_layout)

        # Add banner
        banner_label = QLabel()
        banner_pixmap = QPixmap(os.path.join(CUR_PATH, 'imgs', 'banner1.png'))
        banner_label.setPixmap(
            banner_pixmap.scaled(
                QSize(400, 100),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )
        banner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(banner_label)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        main_layout.addWidget(self.tabs)

        # Create tabs
        self.create_setup_tab()
        self.create_settings_tab()
        self.create_scenery_tab()
        self.create_logs_tab()

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.run_button = StyledButton("Run", primary=True)
        self.run_button.clicked.connect(self.on_run)

        self.save_button = StyledButton("Save")
        self.save_button.clicked.connect(self.on_save)

        self.quit_button = StyledButton("Quit")
        self.quit_button.clicked.connect(self.close)

        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.quit_button)

        main_layout.addLayout(button_layout)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def create_setup_tab(self):
        """Create the setup configuration tab"""
        setup_widget = QWidget()

        # Create scroll area for setup content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Create the actual content widget
        setup_content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        setup_content.setLayout(layout)

        # Paths group
        paths_group = QGroupBox("Paths Configuration")
        paths_layout = QVBoxLayout()
        paths_group.setLayout(paths_layout)

        # Scenery path
        scenery_layout = QHBoxLayout()
        scenery_label = QLabel("Scenery install dir:")
        scenery_label.setToolTip(
            "Directory where AutoOrtho scenery will be installed.\n"
            "This should be a dedicated folder for AutoOrtho scenery,\n"
            "typically separate from your X-Plane Custom Scenery folder.\n"
            "Example: C:\\AutoOrtho_Scenery\\ or ~/AutoOrtho_Scenery/"
        )
        scenery_layout.addWidget(scenery_label)
        self.scenery_path_edit = QLineEdit(self.cfg.paths.scenery_path)
        self.scenery_path_edit.setObjectName('scenery_path')
        self.scenery_path_edit.setToolTip(
            "Full path to your AutoOrtho scenery installation directory"
        )
        scenery_layout.addWidget(self.scenery_path_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(
            lambda: self.browse_folder(self.scenery_path_edit)
        )
        scenery_layout.addWidget(browse_btn)
        paths_layout.addLayout(scenery_layout)

        # X-Plane path
        xplane_layout = QHBoxLayout()
        xplane_label = QLabel("X-Plane install dir:")
        xplane_label.setToolTip(
            "Your main X-Plane installation directory.\n"
            "This should contain the X-Plane.exe file and\n"
            "the 'Custom Scenery' folder.\n"
            "Example: C:\\X-Plane 12\\ or /Applications/X-Plane 12/"
        )
        xplane_layout.addWidget(xplane_label)
        self.xplane_path_edit = QLineEdit(self.cfg.paths.xplane_path)
        self.xplane_path_edit.setObjectName('xplane_path')
        self.xplane_path_edit.setToolTip(
            "Full path to your X-Plane installation directory"
        )
        xplane_layout.addWidget(self.xplane_path_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(
            lambda: self.browse_folder(self.xplane_path_edit)
        )
        xplane_layout.addWidget(browse_btn)
        paths_layout.addLayout(xplane_layout)

        # Cache dir
        cache_layout = QHBoxLayout()
        cache_label = QLabel("Image cache dir:")
        cache_label.setToolTip(
            "Directory for caching downloaded imagery.\n"
            "Should be on a fast drive (SSD recommended) with plenty of "
            "space.\n"
            "Cache helps reduce download times for frequently visited "
            "areas.\n"
            "Optimal: SSD with 50-500GB available space"
        )
        cache_layout.addWidget(cache_label)
        self.cache_dir_edit = QLineEdit(self.cfg.paths.cache_dir)
        self.cache_dir_edit.setObjectName('cache_dir')
        self.cache_dir_edit.setToolTip(
            "Full path to your image cache directory"
        )
        cache_layout.addWidget(self.cache_dir_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(
            lambda: self.browse_folder(self.cache_dir_edit)
        )
        cache_layout.addWidget(browse_btn)
        paths_layout.addLayout(cache_layout)

        # Download dir
        download_layout = QHBoxLayout()
        download_label = QLabel("Temp download dir:")
        download_label.setToolTip(
            "Temporary directory for downloading scenery packages.\n"
            "Should have enough space for large scenery downloads "
            "(10-50GB).\n"
            "Files are deleted after successful installation.\n"
            "Can be on any drive with sufficient free space."
        )
        download_layout.addWidget(download_label)
        self.download_dir_edit = QLineEdit(self.cfg.paths.download_dir)
        self.download_dir_edit.setObjectName('download_dir')
        self.download_dir_edit.setToolTip(
            "Full path to temporary download directory"
        )
        download_layout.addWidget(self.download_dir_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(
            lambda: self.browse_folder(self.download_dir_edit)
        )
        download_layout.addWidget(browse_btn)
        paths_layout.addLayout(download_layout)

        layout.addWidget(paths_group)

        # Options group
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)

        self.showconfig_check = QCheckBox("Always show config menu")
        self.showconfig_check.setChecked(self.cfg.general.showconfig)
        self.showconfig_check.setObjectName('showconfig')
        self.showconfig_check.setToolTip(
            "If enabled, the configuration window will always appear on "
            "startup.\n"
            "If disabled, AutoOrtho will start directly without showing "
            "this window.\n"
            "Recommended: Enabled until you're satisfied with your "
            "configuration."
        )
        options_layout.addWidget(self.showconfig_check)

        # Map type
        maptype_layout = QHBoxLayout()
        maptype_label = QLabel("Map type override:")
        maptype_label.setToolTip(
            "Force AutoOrtho to use a specific imagery source:\n"
            "• BI (Bing): High quality, good worldwide coverage\n"
            "• GO2 (Google): Excellent quality, some restrictions\n"
            "• NAIP: Very high quality for USA only\n"
            "• EOX: Good for Europe and some other regions\n"
            "• USGS: USA government imagery\n"
            "• Firefly: Alternative commercial source\n"
            "Leave empty for automatic selection (recommended)"
        )
        maptype_layout.addWidget(maptype_label)
        self.maptype_combo = QComboBox()
        self.maptype_combo.addItems([
            '', 'BI', 'GO2', 'NAIP', 'EOX', 'USGS', 'Firefly'
        ])
        self.maptype_combo.setCurrentText(self.cfg.autoortho.maptype_override)
        self.maptype_combo.setObjectName('maptype_override')
        self.maptype_combo.setToolTip(
            "Select a specific map provider or leave empty for auto-selection"
        )
        maptype_layout.addWidget(self.maptype_combo)
        maptype_layout.addStretch()
        options_layout.addLayout(maptype_layout)

        self.simheaven_compat_check = QCheckBox("SimHeaven compatibility mode")
        self.simheaven_compat_check.setChecked(self.cfg.autoortho.simheaven_compat)
        self.simheaven_compat_check.setObjectName('simheaven_compat')
        self.simheaven_compat_check.setToolTip(
            "Enable this if you are using SimHeaven scenery.\n"
            "This will disable AutoOrtho Overlays to use the SimHeaven "
            "overlay instead."
        )
        options_layout.addWidget(self.simheaven_compat_check)


        self.simheaven_compat_check.stateChanged.connect(self.on_simheaven_compat_check)

        # Windows specific
        if platform.system() == 'Windows':
            self.winfsp_check = QCheckBox("Prefer WinFSP over Dokan")
            self.winfsp_check.setChecked(self.cfg.windows.prefer_winfsp)
            self.winfsp_check.setObjectName('prefer_winfsp')
            self.winfsp_check.setToolTip(
                "WinFSP generally provides better performance than Dokan.\n"
                "Enable this if you have WinFSP installed.\n"
                "If you experience issues, try disabling this option.\n"
                "Recommended: Enabled (if WinFSP is available)"
            )
            options_layout.addWidget(self.winfsp_check)

        layout.addWidget(options_group)
        layout.addStretch()

        # Set the content widget to the scroll area
        scroll_area.setWidget(setup_content)

        # Create the main layout for the tab
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll_area)
        setup_widget.setLayout(tab_layout)

        self.tabs.addTab(setup_widget, "Setup")

    def create_settings_tab(self):
        """Create the advanced settings configuration tab"""
        settings_widget = QWidget()

        # Create scroll area for settings content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        # Create the actual content widget
        settings_content = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        settings_content.setLayout(layout)

        # Cache settings group
        cache_group = QGroupBox("Cache Settings")
        cache_layout = QVBoxLayout()
        cache_group.setLayout(cache_layout)

        # File cache size
        file_cache_layout = QHBoxLayout()
        file_cache_label = QLabel("File cache size (GB):")
        file_cache_label.setToolTip(
            "Maximum disk space used for caching imagery files.\n"
            "Larger cache = fewer downloads but more disk usage.\n"
            "Optimal: 50-200GB for regular use, 200-500GB for extensive "
            "flying.\n"
            "Minimum recommended: 20GB"
        )
        file_cache_layout.addWidget(file_cache_label)
        self.file_cache_slider = ModernSlider()
        self.file_cache_slider.setRange(10, 500)
        self.file_cache_slider.setSingleStep(5)
        self.file_cache_slider.setValue(
            int(float(self.cfg.cache.file_cache_size))
        )
        self.file_cache_slider.setObjectName('file_cache_size')
        self.file_cache_slider.setToolTip(
            "Drag to adjust maximum cache size in gigabytes"
        )
        self.file_cache_label = QLabel(f"{self.cfg.cache.file_cache_size} GB")
        self.file_cache_slider.valueChanged.connect(
            lambda v: self.file_cache_label.setText(f"{v} GB")
        )
        file_cache_layout.addWidget(self.file_cache_slider)
        file_cache_layout.addWidget(self.file_cache_label)
        self.clean_cache_btn = StyledButton("Clean Cache")
        self.clean_cache_btn.clicked.connect(self.on_clean_cache)
        self.clean_cache_btn.setToolTip(
            "Remove old cached files to free up disk space.\n"
            "This will delete the oldest cached images first."
        )
        file_cache_layout.addWidget(self.clean_cache_btn)
        cache_layout.addLayout(file_cache_layout)

        # Memory cache limit
        mem_cache_layout = QHBoxLayout()
        mem_cache_label = QLabel("Memory cache (GB):")
        mem_cache_label.setToolTip(
            "Maximum RAM used for caching images in memory.\n"
            "Higher values improve performance but use more RAM.\n"
            "Optimal: 4-16GB depending on your system RAM.\n"
            "Don't exceed 25% of your total system RAM."
        )
        mem_cache_layout.addWidget(mem_cache_label)
        self.mem_cache_slider = ModernSlider()
        self.mem_cache_slider.setRange(2, 64)
        self.mem_cache_slider.setValue(
            int(float(self.cfg.cache.cache_mem_limit))
        )
        self.mem_cache_slider.setObjectName('cache_mem_limit')
        self.mem_cache_slider.setToolTip(
            "Drag to adjust maximum memory cache size in gigabytes"
        )
        self.mem_cache_label = QLabel(f"{self.cfg.cache.cache_mem_limit} GB")
        self.mem_cache_slider.valueChanged.connect(
            lambda v: self.mem_cache_label.setText(f"{v} GB")
        )
        mem_cache_layout.addWidget(self.mem_cache_slider)
        mem_cache_layout.addWidget(self.mem_cache_label)
        cache_layout.addLayout(mem_cache_layout)

        layout.addWidget(cache_group)

        # AutoOrtho Settings group
        autoortho_group = QGroupBox("AutoOrtho Settings")
        autoortho_layout = QVBoxLayout()
        autoortho_group.setLayout(autoortho_layout)

        # Min zoom level
        min_zoom_layout = QHBoxLayout()
        min_zoom_label = QLabel("Minimum zoom level:")
        min_zoom_label.setToolTip(
            "Minimum detail level for imagery downloads.\n"
            "Higher values = more detail but larger downloads.\n"
            "8-11: Low detail, faster downloads\n"
            "12-14: Medium detail, balanced (recommended)\n"
            "15-18: High detail, slower downloads\n"
            "Optimal: 13 for most users"
        )
        min_zoom_layout.addWidget(min_zoom_label)
        self.min_zoom_slider = ModernSlider()
        self.min_zoom_slider.setRange(8, 18)
        self.min_zoom_slider.setValue(int(self.cfg.autoortho.min_zoom))
        self.min_zoom_slider.setObjectName('min_zoom')
        self.min_zoom_slider.setToolTip(
            "Drag to adjust minimum zoom level (8=low detail, 18=high detail)"
        )
        self.min_zoom_label = QLabel(f"{self.cfg.autoortho.min_zoom}")
        self.min_zoom_slider.valueChanged.connect(
            lambda v: self.min_zoom_label.setText(f"{v}")
        )
        min_zoom_layout.addWidget(self.min_zoom_slider)
        min_zoom_layout.addWidget(self.min_zoom_label)
        autoortho_layout.addLayout(min_zoom_layout)

        max_zoom_layout = QHBoxLayout()
        max_zoom_label = QLabel("Maximum zoom level:")
        max_zoom_label.setToolTip(
            "Maximum zoom level for imagery downloads.\n"
            "Higher values = more detail but larger downloads and more VRAM usage.\n"
            "Optimal: 16 for most cases. Keep in mind that every extra ZL increases VRAM and potential network usage by 4x."
        )
        max_zoom_layout.addWidget(max_zoom_label)
        self.max_zoom_slider = ModernSlider()
        self.max_zoom_slider.setRange(1, 18)
        self.max_zoom_slider.setValue(int(self.cfg.autoortho.max_zoom))
        self.max_zoom_slider.setObjectName('max_zoom')
        self.max_zoom_slider.setToolTip(
            "Drag to adjust maximum zoom level (1=low detail, 18=high detail)"
        )
        self.max_zoom_label = QLabel(f"{self.cfg.autoortho.max_zoom}")
        self.max_zoom_slider.valueChanged.connect(
            lambda v: self.max_zoom_label.setText(f"{v}")
        )
        max_zoom_layout.addWidget(self.max_zoom_slider)
        max_zoom_layout.addWidget(self.max_zoom_label)
        autoortho_layout.addLayout(max_zoom_layout)

        # Max wait time
        maxwait_layout = QHBoxLayout()
        maxwait_label = QLabel("Max wait time (seconds):")
        maxwait_label.setToolTip(
            "Maximum time to wait for imagery downloads before timing out.\n"
            "Lower values = faster response but may miss imagery\n"
            "Higher values = better reliability but slower response\n"
            "Optimal: 5-15 seconds depending on internet speed\n"
            "Slow internet: 15-30 seconds"
        )
        maxwait_layout.addWidget(maxwait_label)
        self.maxwait_slider = ModernSlider()
        self.maxwait_slider.setRange(1, 100)
        self.maxwait_slider.setSingleStep(1)
        # Convert maxwait to int for slider (multiply by 10 for 0.1 precision)
        maxwait_value = int(float(self.cfg.autoortho.maxwait) * 10)
        self.maxwait_slider.setValue(maxwait_value)
        self.maxwait_slider.setObjectName('maxwait')
        self.maxwait_slider.setToolTip(
            "Drag to adjust maximum wait time in seconds"
        )
        self.maxwait_label = QLabel(f"{self.cfg.autoortho.maxwait}")
        self.maxwait_slider.valueChanged.connect(
            lambda v: self.maxwait_label.setText(f"{v/10:.1f}")
        )
        maxwait_layout.addWidget(self.maxwait_slider)
        maxwait_layout.addWidget(self.maxwait_label)
        autoortho_layout.addLayout(maxwait_layout)

        # Fetch threads
        threads_layout = QHBoxLayout()
        threads_label = QLabel("Fetch threads:")
        threads_label.setToolTip(
            "Number of simultaneous download threads.\n"
            "More threads = faster downloads but higher CPU/network usage.\n"
            "Too many threads may cause timeouts or instability.\n"
            "Optimal: 4-8 threads for most systems\n"
            f"Your system has {os.cpu_count() or 1} CPU threads available."
        )
        threads_layout.addWidget(threads_label)
        self.fetch_threads_spinbox = ModernSpinBox()

        # Get available CPU threads
        max_threads = 128 #os.cpu_count() or 1
        self.fetch_threads_spinbox.setRange(1, max_threads)

        # Ensure initial value doesn't exceed available threads
        initial_threads = min(
            int(self.cfg.autoortho.fetch_threads), max_threads
        )
        self.fetch_threads_spinbox.setValue(initial_threads)
        self.fetch_threads_spinbox.setObjectName('fetch_threads')
        self.fetch_threads_spinbox.setToolTip(
            f"Number of download threads (1-{max_threads})"
        )

        # Add validation when value changes
        #self.fetch_threads_spinbox.valueChanged.connect(self.validate_threads)

        threads_layout.addWidget(self.fetch_threads_spinbox)
        threads_layout.addWidget(QLabel(f"(max: {max_threads})"))
        autoortho_layout.addLayout(threads_layout)

        # Bandwidth limit
        bandwidth_layout = QHBoxLayout()
        bandwidth_label = QLabel("Max bandwidth (Mbits/s):")
        bandwidth_label.setToolTip(
            "Maximum download bandwidth in megabits per second.\n"
            "0 = unlimited bandwidth (default)\n"
            "Useful for limiting network usage during flight\n"
            "or on metered/shared connections.\n"
            "Recommended: 50-200 Mbits for most users"
        )
        bandwidth_layout.addWidget(bandwidth_label)
        self.bandwidth_slider = ModernSlider()
        self.bandwidth_slider.setRange(0, 1000)  # 0 to 1000 Mbits
        self.bandwidth_slider.setSingleStep(10)
        bandwidth_value = int(float(self.cfg.autoortho.max_bandwidth_mbits))
        self.bandwidth_slider.setValue(bandwidth_value)
        self.bandwidth_slider.setObjectName('max_bandwidth_mbits')
        self.bandwidth_slider.setToolTip(
            "Drag to adjust maximum bandwidth limit in megabits per second"
        )
        if bandwidth_value == 0:
            bandwidth_text = "Unlimited"
        else:
            bandwidth_text = f"{bandwidth_value} Mbits/s"
        self.bandwidth_label = QLabel(bandwidth_text)
        self.bandwidth_slider.valueChanged.connect(
            lambda v: self.bandwidth_label.setText(
                "Unlimited" if v == 0 else f"{v} Mbits/s"
            )
        )
        bandwidth_layout.addWidget(self.bandwidth_slider)
        bandwidth_layout.addWidget(self.bandwidth_label)
        autoortho_layout.addLayout(bandwidth_layout)

        layout.addWidget(autoortho_group)

        # DDS Compression Settings group
        dds_group = QGroupBox("DDS Compression Settings")
        dds_layout = QVBoxLayout()
        dds_group.setLayout(dds_layout)

        # Compressor
        compressor_layout = QHBoxLayout()
        compressor_label = QLabel("Compressor:")
        compressor_label.setToolTip(
            "Algorithm used for DDS texture compression:\n"
            "• ISPC: Intel's high-performance compressor (recommended)\n"
            "  - Faster compression, better quality\n"
            "  - Requires modern CPU\n"
            "• STB: Standard compressor (compatibility)\n"
            "  - Slower but works on all systems\n"
            "  - Use if ISPC causes issues"
        )
        compressor_layout.addWidget(compressor_label)
        self.compressor_combo = QComboBox()
        self.compressor_combo.addItems(['ISPC', 'STB'])
        self.compressor_combo.setCurrentText(self.cfg.pydds.compressor)
        self.compressor_combo.setObjectName('compressor')
        self.compressor_combo.setToolTip(
            "Select compression algorithm (ISPC recommended)"
        )
        compressor_layout.addWidget(self.compressor_combo)
        compressor_layout.addStretch()
        dds_layout.addLayout(compressor_layout)

        # Format
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        format_label.setToolTip(
            "DDS compression format:\n"
            "• BC1: Smaller files, no transparency, good for terrain\n"
            "  - 4:1 compression ratio\n"
            "  - Recommended for most scenery\n"
            "• BC3: Larger files, supports transparency\n"
            "  - 3:1 compression ratio\n"
            "  - Use only if transparency is needed"
        )
        format_layout.addWidget(format_label)
        self.format_combo = QComboBox()
        self.format_combo.addItems(['BC1', 'BC3'])
        self.format_combo.setCurrentText(self.cfg.pydds.format)
        self.format_combo.setObjectName('format')
        self.format_combo.setToolTip(
            "Select DDS format (BC1 recommended for most uses)"
        )
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        dds_layout.addLayout(format_layout)

        layout.addWidget(dds_group)

        # General Settings group
        general_group = QGroupBox("General Settings")
        general_layout = QVBoxLayout()
        general_group.setLayout(general_layout)

        self.gui_check = QCheckBox("Use GUI at startup")
        self.gui_check.setChecked(self.cfg.general.gui)
        self.gui_check.setObjectName('gui')
        self.gui_check.setToolTip(
            "Show graphical interface when AutoOrtho starts.\n"
            "If disabled, AutoOrtho runs in background only.\n"
            "Recommended: Enabled for easier monitoring and control."
        )
        general_layout.addWidget(self.gui_check)

        self.hide_check = QCheckBox("Hide window when running")
        self.hide_check.setChecked(self.cfg.general.hide)
        self.hide_check.setObjectName('hide')
        self.hide_check.setToolTip(
            "Minimize AutoOrtho window to system tray when running.\n"
            "Helps keep desktop clean during long flights.\n"
            "You can still access it from the system tray."
        )
        general_layout.addWidget(self.hide_check)

        self.debug_check = QCheckBox("Debug mode")
        self.debug_check.setChecked(self.cfg.general.debug)
        self.debug_check.setObjectName('debug')
        self.debug_check.setToolTip(
            "Enable detailed logging for troubleshooting.\n"
            "Creates larger log files with more information.\n"
            "Only enable if experiencing issues or when asked by support."
        )
        general_layout.addWidget(self.debug_check)

        layout.addWidget(general_group)

        # Scenery Settings group
        scenery_group = QGroupBox("Scenery Settings")
        scenery_layout = QVBoxLayout()
        scenery_group.setLayout(scenery_layout)

        self.noclean_check = QCheckBox("Don't cleanup downloads")
        self.noclean_check.setChecked(self.cfg.scenery.noclean)
        self.noclean_check.setObjectName('noclean')
        self.noclean_check.setToolTip(
            "Keep downloaded scenery files after installation.\n"
            "Useful for reinstalling or sharing scenery packages.\n"
            "Warning: Can use significant disk space over time.\n"
            "Recommended: Disabled unless you need the original files."
        )
        scenery_layout.addWidget(self.noclean_check)

        layout.addWidget(scenery_group)

        # FUSE Settings group
        fuse_group = QGroupBox("FUSE Settings")
        fuse_layout = QVBoxLayout()
        fuse_group.setLayout(fuse_layout)

        self.threading_check = QCheckBox("Enable multi-threading")
        self.threading_check.setChecked(self.cfg.fuse.threading)
        self.threading_check.setObjectName('threading')
        self.threading_check.setToolTip(
            "Use multiple threads for file system operations.\n"
            "Improves performance on multi-core systems.\n"
            "May cause issues on some older systems.\n"
            "Recommended: Enabled on modern multi-core CPUs."
        )
        fuse_layout.addWidget(self.threading_check)

        layout.addWidget(fuse_group)

        # Flight Data Settings group
        flightdata_group = QGroupBox("Flight Data Settings")
        flightdata_layout = QVBoxLayout()
        flightdata_group.setLayout(flightdata_layout)

        # Web UI port
        webui_port_layout = QHBoxLayout()
        webui_port_label = QLabel("Web UI port:")
        webui_port_label.setToolTip(
            "Port number for the web-based monitoring interface.\n"
            "Access via http://localhost:[port] in your browser.\n"
            "Must be an unused port between 1024-65535.\n"
            "Default: 8080. Change if port conflicts occur."
        )
        webui_port_layout.addWidget(webui_port_label)
        self.webui_port_edit = QLineEdit(str(self.cfg.flightdata.webui_port))
        self.webui_port_edit.setObjectName('webui_port')
        self.webui_port_edit.setToolTip(
            "Port number for web interface (default: 8080)"
        )
        webui_port_layout.addWidget(self.webui_port_edit)
        webui_port_layout.addStretch()
        flightdata_layout.addLayout(webui_port_layout)

        # X-Plane UDP port
        xplane_port_layout = QHBoxLayout()
        xplane_port_label = QLabel("X-Plane UDP port:")
        xplane_port_label.setToolTip(
            "UDP port for receiving flight data from X-Plane.\n"
            "Must match the port configured in X-Plane's data output "
            "settings.\n"
            "Default: 49001. Check X-Plane Settings > Data Output."
        )
        xplane_port_layout.addWidget(xplane_port_label)
        self.xplane_udp_port_edit = QLineEdit(
            str(self.cfg.flightdata.xplane_udp_port)
        )
        self.xplane_udp_port_edit.setObjectName('xplane_udp_port')
        self.xplane_udp_port_edit.setToolTip(
            "UDP port for X-Plane data (must match X-Plane settings)"
        )
        xplane_port_layout.addWidget(self.xplane_udp_port_edit)
        xplane_port_layout.addStretch()
        flightdata_layout.addLayout(xplane_port_layout)

        layout.addWidget(flightdata_group)

        layout.addStretch()

        # Set the content widget to the scroll area
        scroll_area.setWidget(settings_content)

        # Create the main layout for the tab
        tab_layout = QVBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(scroll_area)
        settings_widget.setLayout(tab_layout)

        self.tabs.addTab(settings_widget, "Settings")

    def create_scenery_tab(self):
        """Create the scenery management tab"""
        scenery_widget = QWidget()
        layout = QVBoxLayout()
        scenery_widget.setLayout(layout)

        # Create scroll area for scenery list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.scenery_content = QWidget()
        self.scenery_layout = QVBoxLayout()
        self.scenery_content.setLayout(self.scenery_layout)

        scroll_area.setWidget(self.scenery_content)
        layout.addWidget(scroll_area)

        # Refresh scenery list
        self.refresh_scenery_list()

        self.tabs.addTab(scenery_widget, "Scenery")

    def create_logs_tab(self):
        """Create the logs tab"""
        logs_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        logs_widget.setLayout(layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.tabs.addTab(logs_widget, "Logs")

    def refresh_scenery_list(self):
        """Refresh the scenery list display"""
        # Clear existing widgets
        while self.scenery_layout.count():
            child = self.scenery_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.dl.find_regions()

        for r in self.dl.regions.values():
            latest = r.get_latest_release()
            latest.parse()

            # Create scenery item frame
            item_frame = QFrame()
            item_frame.setFrameStyle(QFrame.Shape.Box)
            item_frame.setStyleSheet("""
                QFrame {
                    background-color: #2A2A2A;
                    border: 1px solid #3A3A3A;
                    border-radius: 4px;
                    padding: 10px;
                    margin: 5px;
                }
            """)

            item_layout = QVBoxLayout()
            item_frame.setLayout(item_layout)

            # Title
            title_label = QLabel(f"<b>{latest.name}</b>")
            title_label.setStyleSheet("color: #6eb6ff; font-size: 16px;")
            item_layout.addWidget(title_label)

            pending_update = False
            if r.local_rel:
                version_label = QLabel(f"Current version: {r.local_rel.ver}")
                item_layout.addWidget(version_label)
                if version.parse(latest.ver) > version.parse(r.local_rel.ver):
                    pending_update = True
            else:
                version_label = QLabel("Not installed")
                version_label.setStyleSheet("color: #999;")
                item_layout.addWidget(version_label)
                pending_update = True

            if pending_update:
                info_label = QLabel(
                    f"Available: v{latest.ver} | "
                    f"Size: {latest.totalsize/1048576:.2f} MB | "
                    f"Downloads: {latest.download_count}"
                )
                info_label.setStyleSheet("color: #BBB;")
                item_layout.addWidget(info_label)

                # Progress bar (hidden initially)
                progress_bar = QProgressBar()
                progress_bar.setVisible(False)
                progress_bar.setObjectName(f"progress-{r.region_id}")
                item_layout.addWidget(progress_bar)

                # Install button
                install_btn = StyledButton("Install", primary=True)
                install_btn.setObjectName(f"scenery-{r.region_id}")
                install_btn.clicked.connect(
                    lambda checked, rid=r.region_id: (
                        self.on_install_scenery(rid)
                    )
                )
                item_layout.addWidget(install_btn)
            else:
                status_label = QLabel("✓ Up to date")
                status_label.setStyleSheet("color: #4CAF50;")
                item_layout.addWidget(status_label)

            self.scenery_layout.addWidget(item_frame)

        self.scenery_layout.addStretch()

    def show_splash(self):
        """Show splash screen"""
        splash_pix = QPixmap(os.path.join(CUR_PATH, 'imgs', 'splash.png'))
        self.splash = QSplashScreen(
            splash_pix, Qt.WindowType.WindowStaysOnTopHint
        )
        self.splash.show()
        QTimer.singleShot(2000, self.splash.close)

    def validate_threads(self, value):
        """Validate fetch threads value and show warning if too high"""
        max_threads = os.cpu_count() or 1
        if value > max_threads:
            QMessageBox.information(
                self,
                "Thread Limit",
                f"Number of threads cannot be greater than {max_threads} "
                f"(available CPU threads on this machine).\n"
                f"Value has been adjusted to {max_threads}."
            )
            self.fetch_threads_spinbox.setValue(max_threads)

    def on_simheaven_compat_check(self, state):
        """Handle SimHeaven compatibility check"""
        if state == Qt.CheckState.Checked:
            self.cfg.autoortho.simheaven_compat = True
        else:
            self.cfg.autoortho.simheaven_compat = False

    def browse_folder(self, line_edit):
        """Open folder browser dialog"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Directory",
            line_edit.text()
        )
        if folder:
            line_edit.setText(folder)

    def on_run(self):
        """Handle Run button click"""
        self.save_config()
        self.cfg.load()
        self.update_status_bar("Mounting sceneries...")
        self.mount_sceneries(blocking=False)
        self.verify()
        self.update_status_bar("Running")
        self.showMinimized()

    def on_save(self):
        """Handle Save button click"""
        self.save_config()
        self.cfg.load()
        
        # Update bandwidth limiter with new settings
        try:
            new_bandwidth = float(self.cfg.autoortho.max_bandwidth_mbits)
            getortho.chunk_getter.update_bandwidth_limit(new_bandwidth)
        except (ValueError, AttributeError) as e:
            log.warning(f"Could not update bandwidth limit: {e}")
        
        self.update_status_bar("Configuration saved")

    def on_clean_cache(self):
        """Handle Clean Cache button click"""
        self.update_status_bar("Cleaning cache...")
        self.clean_cache_btn.setEnabled(False)
        self.run_button.setEnabled(False)

        # Run in separate thread
        cache_thread = QThread()
        cache_thread.run = lambda: self.clean_cache(
            self.cfg.paths.cache_dir,
            int(self.file_cache_slider.value())
        )
        cache_thread.finished.connect(lambda: self.on_cache_cleaned())
        cache_thread.start()

    def on_cache_cleaned(self):
        """Called when cache cleaning is complete"""
        self.clean_cache_btn.setEnabled(True)
        self.run_button.setEnabled(True)
        QMessageBox.information(
            self, "Cache Cleaned", "Cache cleaning completed!"
        )

    def on_install_scenery(self, region_id):
        """Handle scenery installation"""
        button = self.findChild(QPushButton, f"scenery-{region_id}")
        progress_bar = self.findChild(QProgressBar, f"progress-{region_id}")

        if button:
            button.setEnabled(False)
            button.setText("Working...")

        if progress_bar:
            progress_bar.setVisible(True)

        # Create worker thread
        worker = SceneryDownloadWorker(
            self.dl, region_id, self.cfg.paths.download_dir
        )
        worker.progress.connect(self.on_download_progress)
        worker.finished.connect(self.on_download_finished)
        worker.error.connect(self.on_download_error)

        self.download_workers[region_id] = worker
        worker.start()

    def on_download_progress(self, region_id, progress_data):
        """Update download progress"""
        progress_bar = self.findChild(QProgressBar, f"progress-{region_id}")
        if progress_bar:
            pcnt_done = progress_data.get('pcnt_done', 0)
            progress_bar.setValue(int(pcnt_done))

        status = progress_data.get('status', 'Downloading...')
        MBps = progress_data.get('MBps', 0)
        if pcnt_done > 0:
            self.update_status_bar(
                f"{region_id}: {pcnt_done:.1f}% ({MBps:.1f} MB/s)"
            )
        else:
            self.update_status_bar(f"{region_id}: {status}")

    def on_download_finished(self, region_id, success):
        """Handle download completion"""
        button = self.findChild(QPushButton, f"scenery-{region_id}")
        progress_bar = self.findChild(QProgressBar, f"progress-{region_id}")

        if success:
            if button:
                button.setVisible(False)
            if progress_bar:
                progress_bar.setVisible(False)
            self.update_status_bar(f"Successfully installed {region_id}")
            # Refresh the scenery list
            self.refresh_scenery_list()
        else:
            if button:
                button.setText("Retry?")
                button.setEnabled(True)
            if progress_bar:
                progress_bar.setVisible(False)
            self.update_status_bar(f"Failed to install {region_id}")

        # Clean up worker
        if region_id in self.download_workers:
            del self.download_workers[region_id]

    def on_download_error(self, region_id, error_msg):
        """Handle download error"""
        self.show_error.emit(f"Failed to install {region_id}:\n{error_msg}")
        self.on_download_finished(region_id, False)

    def save_config(self):
        """Save configuration from UI to config object"""
        self.ready.clear()

        # Save paths
        self.cfg.paths.scenery_path = self.scenery_path_edit.text()
        self.cfg.paths.xplane_path = self.xplane_path_edit.text()
        self.cfg.paths.cache_dir = self.cache_dir_edit.text()
        self.cfg.paths.download_dir = self.download_dir_edit.text()

        # Save options
        self.cfg.general.showconfig = self.showconfig_check.isChecked()
        self.cfg.autoortho.maptype_override = self.maptype_combo.currentText()
        self.cfg.autoortho.simheaven_compat = self.simheaven_compat_check.isChecked()

        # Windows specific
        if platform.system() == 'Windows' and hasattr(self, 'winfsp_check'):
            self.cfg.windows.prefer_winfsp = self.winfsp_check.isChecked()

        # Save Settings tab values
        if hasattr(self, 'file_cache_slider'):
            # Cache settings
            self.cfg.cache.file_cache_size = str(
                self.file_cache_slider.value()
            )
            self.cfg.cache.cache_mem_limit = str(
                self.mem_cache_slider.value()
            )

            # AutoOrtho settings
            self.cfg.autoortho.min_zoom = str(self.min_zoom_slider.value())
            self.cfg.autoortho.max_zoom = str(self.max_zoom_slider.value())
            self.cfg.autoortho.maxwait = str(
                self.maxwait_slider.value() / 10.0
            )
            self.cfg.autoortho.fetch_threads = str(
                self.fetch_threads_spinbox.value()
            )
            self.cfg.autoortho.max_bandwidth_mbits = str(
                self.bandwidth_slider.value()
            )

            # DDS settings
            self.cfg.pydds.compressor = self.compressor_combo.currentText()
            self.cfg.pydds.format = self.format_combo.currentText()

            # General settings
            self.cfg.general.gui = self.gui_check.isChecked()
            self.cfg.general.hide = self.hide_check.isChecked()
            self.cfg.general.debug = self.debug_check.isChecked()

            # Scenery settings
            self.cfg.scenery.noclean = self.noclean_check.isChecked()

            # FUSE settings
            self.cfg.fuse.threading = self.threading_check.isChecked()

            # Flight data settings
            self.cfg.flightdata.webui_port = str(
                self.webui_port_edit.text()
            )
            self.cfg.flightdata.xplane_udp_port = str(
                self.xplane_udp_port_edit.text()
            )

        self.cfg.save()
        self.ready.set()
        self.refresh_scenery()

    def refresh_scenery(self):
        """Refresh scenery data"""
        self.dl.regions = {}
        self.dl.extract_dir = self.cfg.paths.scenery_path
        self.dl.download_dir = self.cfg.paths.download_dir
        self.dl.find_regions()
        for r in self.dl.regions.values():
            latest = r.get_latest_release()
            latest.parse()

    def update_logs(self):
        """Update log display"""
        try:
            if os.path.exists(self.cfg.paths.log_file):
                with open(self.cfg.paths.log_file) as h:
                    lines = h.readlines()[-50:]  # Last 50 lines
                    self.log_text.setPlainText(''.join(lines))
                    # Auto scroll to bottom
                    scrollbar = self.log_text.verticalScrollBar()
                    scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass

    def update_status_bar(self, message):
        """Update status bar message"""
        self.status_bar.showMessage(message)
        log.info(message)

    def append_log(self, message):
        """Append message to log display"""
        self.log_text.append(message)

    def display_error(self, message):
        """Display error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def verify(self):
        """Verify configuration"""
        self._check_xplane_dir(self.cfg.paths.xplane_path)
        for scenery in self.cfg.scenery_mounts:
            self._check_ortho_dir(scenery.get('root'))

        if not self.cfg.scenery_mounts:
            self.errors.append("No installed scenery detected!")

        msg = []
        if self.warnings:
            msg.append("WARNINGS:")
            msg.extend(self.warnings)
            msg.append("\n")

        for warn in self.warnings:
            log.warning(warn)

        if self.errors:
            msg.append("ERRORS:")
            msg.extend(self.errors)
            msg.append("\nWILL EXIT DUE TO ERRORS")

        for err in self.errors:
            log.error(err)

        if msg:
            if self.cfg.general.gui:
                QMessageBox.warning(
                    self, "Configuration Issues", "\n".join(msg)
                )

        if self.errors:
            log.error("ERRORS DETECTED. Exiting.")
            sys.exit(1)

    def clean_cache(self, cache_dir, size_gb):
        """Clean cache directory"""
        self.status_update.emit(
            f"Cleaning up cache_dir {cache_dir}. Please wait..."
        )

        target_gb = max(size_gb, 10)
        target_bytes = pow(2, 30) * target_gb

        try:
            cfiles = sorted(
                pathlib.Path(cache_dir).glob('**/*'), key=os.path.getmtime
            )
            if not cfiles:
                self.status_update.emit("Cache is empty.")
                return

            cache_bytes = sum(
                file.stat().st_size for file in cfiles if file.is_file()
            )
            cachecount = len(cfiles)
            avgcachesize = cache_bytes / cachecount if cachecount > 0 else 0

            self.status_update.emit(
                f"Cache has {cachecount} files. "
                f"Total size approx {cache_bytes//1048576} MB."
            )

            empty_files = [
                x for x in cfiles if x.is_file() and x.stat().st_size == 0
            ]
            self.status_update.emit(
                f"Found {len(empty_files)} empty files to cleanup."
            )
            for file in empty_files:
                if os.path.exists(file):
                    os.remove(file)

            if target_bytes > cache_bytes:
                self.status_update.emit("Cache within size limits.")
                return

            to_delete = int((cache_bytes - target_bytes) // avgcachesize)

            self.status_update.emit(
                f"Over cache size limit, will remove {to_delete} files."
            )
            for file in cfiles[:to_delete]:
                if file.is_file():
                    os.remove(file)

            self.status_update.emit("Cache cleanup done.")
        except Exception as e:
            self.status_update.emit(f"Cache cleanup error: {str(e)}")

    def _check_ortho_dir(self, path):
        """Check if orthophoto directory is valid"""
        ret = True
        if not sorted(pathlib.Path(path).glob("Earth nav data/*/*.dsf")):
            self.warnings.append(
                f"Orthophoto dir {path} seems wrong. "
                "This may cause issues."
            )
            ret = False
        return ret

    def _check_xplane_dir(self, path):
        """Check if X-Plane directory is valid"""
        if not os.path.isdir(path):
            self.errors.append(
                f"XPlane install directory '{path}' is not a directory."
            )
            return False

        if "Custom Scenery" not in os.listdir(path):
            self.errors.append(
                f"XPlane install directory '{path}' seems wrong."
            )
            return False

        return True

    def closeEvent(self, event):
        """Handle window close event"""
        self.running = False
        if hasattr(self, 'log_timer'):
            self.log_timer.stop()
        # Stop all download workers
        for worker in self.download_workers.values():
            worker.terminate()
            worker.wait()
        # Clean up background mount processes
        if hasattr(self, 'unmount_sceneries'):
            self.unmount_sceneries()
        event.accept()


# This class needs to be imported from the parent module
# We'll create a stub here and modify autoortho.py to use the Qt version
class AOMountUI(ConfigUI):
    """Combined UI and mount functionality"""
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mount_threads = []
        self.mounts_running = False

    def mount_sceneries(self, blocking=True):
        """Mount sceneries (stub - implemented in parent)"""
        pass

    def unmount_sceneries(self):
        """Unmount sceneries (stub - implemented in parent)"""
        pass
