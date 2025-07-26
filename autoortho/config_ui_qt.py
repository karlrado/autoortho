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
    QSplashScreen, QGroupBox, QProgressBar, QStatusBar, QFrame
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize
)
from PyQt6.QtGui import (
    QPixmap, QIcon
)

import downloader
from version import __version__

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
                    background-color: #FF6B35;
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
                    border-color: #FF6B35;
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
                background: #FF6B35;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #FF8555;
            }
            QSlider::sub-page:horizontal {
                background: #FF6B35;
                border-radius: 3px;
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
                color: #FF6B35;
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
                border-color: #FF6B35;
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
                background-color: #FF6B35;
                border-color: #FF6B35;
            }
            QComboBox {
                background-color: #3A3A3A;
                border: 1px solid #555;
                padding: 6px;
                border-radius: 4px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #FF6B35;
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
                color: #FF6B35;
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
        banner_label.setPixmap(banner_pixmap.scaled(QSize(400, 100), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
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
        layout = QVBoxLayout()
        layout.setSpacing(15)
        setup_widget.setLayout(layout)

        # Paths group
        paths_group = QGroupBox("Paths Configuration")
        paths_layout = QVBoxLayout()
        paths_group.setLayout(paths_layout)

        # Scenery path
        scenery_layout = QHBoxLayout()
        scenery_layout.addWidget(QLabel("Scenery install dir:"))
        self.scenery_path_edit = QLineEdit(self.cfg.paths.scenery_path)
        self.scenery_path_edit.setObjectName('scenery_path')
        scenery_layout.addWidget(self.scenery_path_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_folder(self.scenery_path_edit))
        scenery_layout.addWidget(browse_btn)
        paths_layout.addLayout(scenery_layout)

        # X-Plane path
        xplane_layout = QHBoxLayout()
        xplane_layout.addWidget(QLabel("X-Plane install dir:"))
        self.xplane_path_edit = QLineEdit(self.cfg.paths.xplane_path)
        self.xplane_path_edit.setObjectName('xplane_path')
        xplane_layout.addWidget(self.xplane_path_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_folder(self.xplane_path_edit))
        xplane_layout.addWidget(browse_btn)
        paths_layout.addLayout(xplane_layout)

        # Cache dir
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("Image cache dir:"))
        self.cache_dir_edit = QLineEdit(self.cfg.paths.cache_dir)
        self.cache_dir_edit.setObjectName('cache_dir')
        cache_layout.addWidget(self.cache_dir_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_folder(self.cache_dir_edit))
        cache_layout.addWidget(browse_btn)
        paths_layout.addLayout(cache_layout)

        # Download dir
        download_layout = QHBoxLayout()
        download_layout.addWidget(QLabel("Temp download dir:"))
        self.download_dir_edit = QLineEdit(self.cfg.paths.download_dir)
        self.download_dir_edit.setObjectName('download_dir')
        download_layout.addWidget(self.download_dir_edit)
        browse_btn = StyledButton("Browse")
        browse_btn.clicked.connect(lambda: self.browse_folder(self.download_dir_edit))
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
        options_layout.addWidget(self.showconfig_check)

        # Map type
        maptype_layout = QHBoxLayout()
        maptype_layout.addWidget(QLabel("Map type override:"))
        self.maptype_combo = QComboBox()
        self.maptype_combo.addItems(['', 'BI', 'GO2', 'NAIP', 'EOX', 'USGS', 'Firefly'])
        self.maptype_combo.setCurrentText(self.cfg.autoortho.maptype_override)
        self.maptype_combo.setObjectName('maptype_override')
        maptype_layout.addWidget(self.maptype_combo)
        maptype_layout.addStretch()
        options_layout.addLayout(maptype_layout)

        # Windows specific
        if platform.system() == 'Windows':
            self.winfsp_check = QCheckBox("Prefer WinFSP over Dokan")
            self.winfsp_check.setChecked(self.cfg.windows.prefer_winfsp)
            self.winfsp_check.setObjectName('prefer_winfsp')
            options_layout.addWidget(self.winfsp_check)

        layout.addWidget(options_group)
        layout.addStretch()

        self.tabs.addTab(setup_widget, "Setup")

    def create_settings_tab(self):
        """Create the advanced settings configuration tab"""
        settings_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15)
        settings_widget.setLayout(layout)

        # Cache settings group
        cache_group = QGroupBox("Cache Settings")
        cache_layout = QVBoxLayout()
        cache_group.setLayout(cache_layout)

        # File cache size
        file_cache_layout = QHBoxLayout()
        file_cache_layout.addWidget(QLabel("File cache size (GB):"))
        self.file_cache_slider = ModernSlider()
        self.file_cache_slider.setRange(10, 500)
        self.file_cache_slider.setSingleStep(5)
        self.file_cache_slider.setValue(int(float(self.cfg.cache.file_cache_size)))
        self.file_cache_slider.setObjectName('file_cache_size')
        self.file_cache_label = QLabel(f"{self.cfg.cache.file_cache_size} GB")
        self.file_cache_slider.valueChanged.connect(
            lambda v: self.file_cache_label.setText(f"{v} GB")
        )
        file_cache_layout.addWidget(self.file_cache_slider)
        file_cache_layout.addWidget(self.file_cache_label)
        self.clean_cache_btn = StyledButton("Clean Cache")
        self.clean_cache_btn.clicked.connect(self.on_clean_cache)
        file_cache_layout.addWidget(self.clean_cache_btn)
        cache_layout.addLayout(file_cache_layout)

        # Memory cache limit
        mem_cache_layout = QHBoxLayout()
        mem_cache_layout.addWidget(QLabel("Memory cache (GB):"))
        self.mem_cache_slider = ModernSlider()
        self.mem_cache_slider.setRange(2, 64)
        self.mem_cache_slider.setValue(int(float(self.cfg.cache.cache_mem_limit)))
        self.mem_cache_slider.setObjectName('cache_mem_limit')
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
        min_zoom_layout.addWidget(QLabel("Minimum zoom level:"))
        self.min_zoom_slider = ModernSlider()
        self.min_zoom_slider.setRange(8, 18)
        self.min_zoom_slider.setValue(int(self.cfg.autoortho.min_zoom))
        self.min_zoom_slider.setObjectName('min_zoom')
        self.min_zoom_label = QLabel(f"{self.cfg.autoortho.min_zoom}")
        self.min_zoom_slider.valueChanged.connect(
            lambda v: self.min_zoom_label.setText(f"{v}")
        )
        min_zoom_layout.addWidget(self.min_zoom_slider)
        min_zoom_layout.addWidget(self.min_zoom_label)
        autoortho_layout.addLayout(min_zoom_layout)

        # Max wait time
        maxwait_layout = QHBoxLayout()
        maxwait_layout.addWidget(QLabel("Max wait time (seconds):"))
        self.maxwait_slider = ModernSlider()
        self.maxwait_slider.setRange(1, 100)
        self.maxwait_slider.setSingleStep(1)
        # Convert maxwait to int for slider (multiply by 10 for 0.1 precision)
        maxwait_value = int(float(self.cfg.autoortho.maxwait) * 10)
        self.maxwait_slider.setValue(maxwait_value)
        self.maxwait_slider.setObjectName('maxwait')
        self.maxwait_label = QLabel(f"{self.cfg.autoortho.maxwait}")
        self.maxwait_slider.valueChanged.connect(
            lambda v: self.maxwait_label.setText(f"{v/10:.1f}")
        )
        maxwait_layout.addWidget(self.maxwait_slider)
        maxwait_layout.addWidget(self.maxwait_label)
        autoortho_layout.addLayout(maxwait_layout)

        # Fetch threads
        threads_layout = QHBoxLayout()
        threads_layout.addWidget(QLabel("Fetch threads:"))
        self.fetch_threads_slider = ModernSlider()
        self.fetch_threads_slider.setRange(1, 64)
        self.fetch_threads_slider.setValue(int(self.cfg.autoortho.fetch_threads))
        self.fetch_threads_slider.setObjectName('fetch_threads')
        self.fetch_threads_label = QLabel(f"{self.cfg.autoortho.fetch_threads}")
        self.fetch_threads_slider.valueChanged.connect(
            lambda v: self.fetch_threads_label.setText(f"{v}")
        )
        threads_layout.addWidget(self.fetch_threads_slider)
        threads_layout.addWidget(self.fetch_threads_label)
        autoortho_layout.addLayout(threads_layout)

        layout.addWidget(autoortho_group)

        # DDS Compression Settings group
        dds_group = QGroupBox("DDS Compression Settings")
        dds_layout = QVBoxLayout()
        dds_group.setLayout(dds_layout)

        # Compressor
        compressor_layout = QHBoxLayout()
        compressor_layout.addWidget(QLabel("Compressor:"))
        self.compressor_combo = QComboBox()
        self.compressor_combo.addItems(['ISPC', 'STB'])
        self.compressor_combo.setCurrentText(self.cfg.pydds.compressor)
        self.compressor_combo.setObjectName('compressor')
        compressor_layout.addWidget(self.compressor_combo)
        compressor_layout.addStretch()
        dds_layout.addLayout(compressor_layout)

        # Format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['BC1', 'BC3'])
        self.format_combo.setCurrentText(self.cfg.pydds.format)
        self.format_combo.setObjectName('format')
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
        general_layout.addWidget(self.gui_check)

        self.hide_check = QCheckBox("Hide window when running")
        self.hide_check.setChecked(self.cfg.general.hide)
        self.hide_check.setObjectName('hide')
        general_layout.addWidget(self.hide_check)

        self.debug_check = QCheckBox("Debug mode")
        self.debug_check.setChecked(self.cfg.general.debug)
        self.debug_check.setObjectName('debug')
        general_layout.addWidget(self.debug_check)

        layout.addWidget(general_group)

        # Scenery Settings group
        scenery_group = QGroupBox("Scenery Settings")
        scenery_layout = QVBoxLayout()
        scenery_group.setLayout(scenery_layout)

        self.noclean_check = QCheckBox("Don't cleanup downloads")
        self.noclean_check.setChecked(self.cfg.scenery.noclean)
        self.noclean_check.setObjectName('noclean')
        scenery_layout.addWidget(self.noclean_check)

        layout.addWidget(scenery_group)

        # FUSE Settings group
        fuse_group = QGroupBox("FUSE Settings")
        fuse_layout = QVBoxLayout()
        fuse_group.setLayout(fuse_layout)

        self.threading_check = QCheckBox("Enable multi-threading")
        self.threading_check.setChecked(self.cfg.fuse.threading)
        self.threading_check.setObjectName('threading')
        fuse_layout.addWidget(self.threading_check)

        layout.addWidget(fuse_group)

        # Flight Data Settings group
        flightdata_group = QGroupBox("Flight Data Settings")
        flightdata_layout = QVBoxLayout()
        flightdata_group.setLayout(flightdata_layout)

        # Web UI port
        webui_port_layout = QHBoxLayout()
        webui_port_layout.addWidget(QLabel("Web UI port:"))
        self.webui_port_edit = QLineEdit(str(self.cfg.flightdata.webui_port))
        self.webui_port_edit.setObjectName('webui_port')
        webui_port_layout.addWidget(self.webui_port_edit)
        webui_port_layout.addStretch()
        flightdata_layout.addLayout(webui_port_layout)

        # X-Plane UDP port
        xplane_port_layout = QHBoxLayout()
        xplane_port_layout.addWidget(QLabel("X-Plane UDP port:"))
        self.xplane_udp_port_edit = QLineEdit(str(self.cfg.flightdata.xplane_udp_port))
        self.xplane_udp_port_edit.setObjectName('xplane_udp_port')
        xplane_port_layout.addWidget(self.xplane_udp_port_edit)
        xplane_port_layout.addStretch()
        flightdata_layout.addLayout(xplane_port_layout)

        layout.addWidget(flightdata_group)

        layout.addStretch()

        self.tabs.addTab(settings_widget, "Settings")

    def create_scenery_tab(self):
        """Create the scenery management tab"""
        scenery_widget = QWidget()
        layout = QVBoxLayout()
        scenery_widget.setLayout(layout)

        # Create scroll area for scenery list
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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
            title_label.setStyleSheet("color: #FF6B35; font-size: 16px;")
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
                install_btn.clicked.connect(lambda checked, rid=r.region_id: self.on_install_scenery(rid))
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
        self.splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
        self.splash.show()
        QTimer.singleShot(2000, self.splash.close)

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
        QMessageBox.information(self, "Cache Cleaned", "Cache cleaning completed!")

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
        worker = SceneryDownloadWorker(self.dl, region_id, self.cfg.paths.download_dir)
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
            self.update_status_bar(f"{region_id}: {pcnt_done:.1f}% ({MBps:.1f} MB/s)")
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

        # Windows specific
        if platform.system() == 'Windows' and hasattr(self, 'winfsp_check'):
            self.cfg.windows.prefer_winfsp = self.winfsp_check.isChecked()

        # Save Settings tab values
        if hasattr(self, 'file_cache_slider'):
            # Cache settings
            self.cfg.cache.file_cache_size = str(self.file_cache_slider.value())
            self.cfg.cache.cache_mem_limit = str(self.mem_cache_slider.value())

            # AutoOrtho settings
            self.cfg.autoortho.min_zoom = str(self.min_zoom_slider.value())
            self.cfg.autoortho.maxwait = str(self.maxwait_slider.value() / 10.0)
            self.cfg.autoortho.fetch_threads = str(self.fetch_threads_slider.value())

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
            self.cfg.flightdata.webui_port = str(self.webui_port_edit.text())
            self.cfg.flightdata.xplane_udp_port = str(self.xplane_udp_port_edit.text())

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
                QMessageBox.warning(self, "Configuration Issues", "\n".join(msg))

        if self.errors:
            log.error("ERRORS DETECTED. Exiting.")
            sys.exit(1)

    def clean_cache(self, cache_dir, size_gb):
        """Clean cache directory"""
        self.status_update.emit(f"Cleaning up cache_dir {cache_dir}. Please wait...")

        target_gb = max(size_gb, 10)
        target_bytes = pow(2, 30) * target_gb

        try:
            cfiles = sorted(pathlib.Path(cache_dir).glob('**/*'), key=os.path.getmtime)
            if not cfiles:
                self.status_update.emit("Cache is empty.")
                return

            cache_bytes = sum(file.stat().st_size for file in cfiles if file.is_file())
            cachecount = len(cfiles)
            avgcachesize = cache_bytes / cachecount if cachecount > 0 else 0

            self.status_update.emit(f"Cache has {cachecount} files. Total size approx {cache_bytes//1048576} MB.")

            empty_files = [x for x in cfiles if x.is_file() and x.stat().st_size == 0]
            self.status_update.emit(f"Found {len(empty_files)} empty files to cleanup.")
            for file in empty_files:
                if os.path.exists(file):
                    os.remove(file)

            if target_bytes > cache_bytes:
                self.status_update.emit("Cache within size limits.")
                return

            to_delete = int((cache_bytes - target_bytes) // avgcachesize)

            self.status_update.emit(f"Over cache size limit, will remove {to_delete} files.")
            for file in cfiles[:to_delete]:
                if file.is_file():
                    os.remove(file)

            self.status_update.emit("Cache cleanup done.")
        except Exception as e:
            self.status_update.emit(f"Cache cleanup error: {str(e)}")

    def _check_ortho_dir(self, path):
        """Check if orthophoto directory is valid"""
        ret = True
        if not sorted(pathlib.Path(path).glob(f"Earth nav data/*/*.dsf")):
            self.warnings.append(f"Orthophoto dir {path} seems wrong. This may cause issues.")
            ret = False
        return ret

    def _check_xplane_dir(self, path):
        """Check if X-Plane directory is valid"""
        if not os.path.isdir(path):
            self.errors.append(f"XPlane install directory '{path}' is not a directory.")
            return False

        if "Custom Scenery" not in os.listdir(path):
            self.errors.append(f"XPlane install directory '{path}' seems wrong.")
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
