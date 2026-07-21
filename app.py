"""
app.py -- Smart Traffic Signal Controller: Desktop Simulation UI

A PyQt5 desktop application that simulates a 4-way intersection with an
AI-powered signal controller. Features an animated intersection view,
live KPI charts, Claude LLM-powered explanations with chat, and a
baseline comparison mode.

Usage:
    python app.py

Environment:
    GEMINI_API_KEY  (optional) -- enables Gemini LLM explanations
"""

import sys
import os
import json
import math
import time
import re
from functools import partial
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QComboBox, QSlider, QSpinBox,
    QCheckBox, QTextEdit, QLineEdit, QTableWidget, QTableWidgetItem,
    QSplitter, QFrame, QGroupBox, QStatusBar, QDialog, QProgressBar,
    QScrollArea, QSizePolicy, QMessageBox, QHeaderView, QFileDialog,
    QTabWidget
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, pyqtSlot, QRectF, QPointF, QPropertyAnimation,
    QEasingCurve, QObject
)
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPainterPath, QIcon, QPalette, QFontDatabase
)
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Qt5Agg')

from traffic_engine import (
    Intersection, Lane, Vehicle, SignalPhase, PhaseDecision,
    PriorityOptimizer, FixedTimerController, SimulationRunner,
    MetricsCollector, SimConstraints, generate_scenario_data,
    load_preset_scenarios, load_csv_scenario
)
from ai_explainer import GeminiExplainer, TemplateFallback


# Set your Gemini API Key here (or set the GEMINI_API_KEY environment variable)
API_KEY = "your-api-key-here"


# ────────────────────────────────────────────────────────────
# Color Palette & Styling (Cyberpunk / Tron HUD theme)
# ────────────────────────────────────────────────────────────

COLORS = {
    "bg_dark": "#05050a",
    "bg_panel": "#0c0e17",
    "bg_card": "#121420",
    "bg_input": "#1b1e2e",
    "accent_primary": "#00ffcc",     # Neon turquoise/cyan
    "accent_secondary": "#ff007f",   # Neon hot pink/magenta
    "accent_green": "#39ff14",       # Neon green
    "accent_red": "#ff3131",         # Neon red
    "accent_yellow": "#ffdf00",      # Neon yellow
    "accent_orange": "#ff5e00",      # Neon orange
    "text_primary": "#e2e8f0",       # Clean slate white
    "text_secondary": "#94a3b8",     # Slate gray
    "text_muted": "#475569",         # Dark slate
    "border": "#1b233a",             # Tech border
    "road": "#101423",               # Dark pavement
    "road_marking": "#00ffcc",       # Glowing dashed lines
    "vehicle_normal": "#ff007f",     # Hot pink normal vehicles
    "vehicle_emergency": "#ff3131",  # Red emergency vehicles
    "vehicle_cleared": "#39ff14",    # Green cleared vehicles
    "signal_green": "#39ff14",
    "signal_yellow": "#ffdf00",
    "signal_red": "#ff3131",
    "crosswalk": "#22314d",
}

DARK_STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}
QWidget {{
    background-color: transparent;
    color: {COLORS['text_primary']};
    font-family: 'Consolas', 'Segoe UI', 'Inter', monospace;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {COLORS['bg_panel']};
    border: 2px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    font-size: 13px;
    color: {COLORS['accent_primary']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: {COLORS['accent_primary']};
}}
QPushButton {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 2px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
}}
QPushButton:hover {{
    background-color: transparent;
    color: {COLORS['accent_primary']};
    border-color: {COLORS['accent_primary']};
}}
QPushButton:pressed {{
    background-color: {COLORS['accent_secondary']};
    color: {COLORS['bg_dark']};
    border-color: {COLORS['accent_secondary']};
}}
QPushButton:disabled {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_muted']};
    border-color: {COLORS['bg_dark']};
}}
QComboBox {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 120px;
}}
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_primary']};
    selection-color: {COLORS['bg_dark']};
}}
QSlider::groove:horizontal {{
    border: none;
    height: 4px;
    background: {COLORS['bg_input']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['accent_primary']};
    border: 1px solid {COLORS['bg_dark']};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {COLORS['accent_primary']};
    border-radius: 2px;
}}
QSpinBox {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 4px 8px;
}}
QCheckBox {{
    color: {COLORS['text_primary']};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid {COLORS['border']};
    border-radius: 3px;
    background-color: {COLORS['bg_input']};
}}
QCheckBox::indicator:checked {{
    background-color: {COLORS['accent_primary']};
    border-color: {COLORS['accent_primary']};
}}
QTextEdit {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px;
    font-size: 12px;
}}
QLineEdit {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px 12px;
}}
QLineEdit:focus {{
    border-color: {COLORS['accent_primary']};
}}
QTableWidget {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    gridline-color: {COLORS['border']};
    font-size: 12px;
}}
QTableWidget::item {{
    padding: 4px 8px;
}}
QHeaderView::section {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['accent_primary']};
    border: 1px solid {COLORS['border']};
    padding: 4px 8px;
    font-weight: bold;
    font-size: 11px;
}}
QStatusBar {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['text_secondary']};
    border-top: 2px solid {COLORS['border']};
    font-size: 12px;
}}
QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background-color: {COLORS['border']};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QProgressBar {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    text-align: center;
    color: {COLORS['text_primary']};
    font-size: 11px;
    font-weight: bold;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent_primary']};
    border-radius: 3px;
}}
QLabel {{
    color: {COLORS['text_primary']};
}}
QTabWidget::pane {{
    border: 2px solid {COLORS['border']};
    background-color: {COLORS['bg_panel']};
    border-radius: 8px;
    margin-top: -2px;
}}
QTabBar::tab {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_secondary']};
    border: 2px solid {COLORS['border']};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 12px;
    text-transform: uppercase;
    margin-right: 2px;
}}
QTabBar::tab:hover {{
    color: {COLORS['accent_primary']};
    border-color: {COLORS['accent_primary']};
}}
QTabBar::tab:selected {{
    background-color: {COLORS['bg_panel']};
    color: {COLORS['accent_primary']};
    border-color: {COLORS['accent_primary']};
    border-bottom: 2px solid {COLORS['bg_panel']};
}}
"""


# ────────────────────────────────────────────────────────────
# Intersection Canvas (QGraphicsView)
# ────────────────────────────────────────────────────────────

class IntersectionCanvas(QGraphicsView):
    """
    Animated top-down 2D view of a 4-way intersection.
    Vehicles are colored ellipses queuing on approaches;
    traffic lights change color live.
    """

    CANVAS_SIZE = 500
    ROAD_WIDTH = 90
    LANE_WIDTH = 40
    VEHICLE_SIZE = 14
    VEHICLE_GAP = 18
    LIGHT_RADIUS = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setStyleSheet(f"background-color: {COLORS['bg_dark']}; border: none;")
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._vehicles: dict[int, QGraphicsEllipseItem] = {}
        self._lights: dict[str, list] = {}  # direction -> [green, yellow, red] items

        self._current_phase = SignalPhase.NORTH_SOUTH_GREEN
        self._queues = {"north": 0, "south": 0, "east": 0, "west": 0}

        self._draw_intersection()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _draw_intersection(self):
        """Draw the static intersection with Cyberpunk Tron grid markings and glow roads."""
        self.scene.clear()
        self._vehicles.clear()
        self._lights.clear()

        S = self.CANVAS_SIZE
        R = self.ROAD_WIDTH
        center = S / 2

        # Background
        self.scene.setSceneRect(0, 0, S, S)
        self.scene.setBackgroundBrush(QBrush(QColor(COLORS["bg_dark"])))

        # Sidewalk areas corners painted as dark tech panels
        corners = [
            (0, 0, center - R/2, center - R/2),
            (center + R/2, 0, center - R/2, center - R/2),
            (0, center + R/2, center - R/2, center - R/2),
            (center + R/2, center + R/2, center - R/2, center - R/2),
        ]
        for x, y, w, h in corners:
            # Draw sidewalk rectangles with glowing border
            self.scene.addRect(x, y, w, h, QPen(QColor(COLORS["border"]), 1.5), QBrush(QColor("#080911")))

        # Subtle tactical HUD coordinate dot grid overlay on sidewalks
        grid_brush = QBrush(QColor(COLORS["border"]))
        for gx in range(12, S, 24):
            for gy in range(12, S, 24):
                if abs(gx - center) > R/2 and abs(gy - center) > R/2:
                    self.scene.addEllipse(gx - 1, gy - 1, 2, 2, QPen(Qt.NoPen), grid_brush)

        # Roads (vertical and horizontal)
        road_color = QColor(COLORS["road"])
        road_pen = QPen(Qt.NoPen)

        # Vertical road (North-South)
        self.scene.addRect(center - R/2, 0, R, S, road_pen, QBrush(road_color))
        # Horizontal road (East-West)
        self.scene.addRect(0, center - R/2, S, R, road_pen, QBrush(road_color))

        # Glowing Neon road boundaries
        glow_pen = QPen(QColor(COLORS["accent_primary"]), 2)
        # North road borders
        self.scene.addLine(center - R/2, 0, center - R/2, center - R/2, glow_pen)
        self.scene.addLine(center + R/2, 0, center + R/2, center - R/2, glow_pen)
        # South road borders
        self.scene.addLine(center - R/2, center + R/2, center - R/2, S, glow_pen)
        self.scene.addLine(center + R/2, center + R/2, center + R/2, S, glow_pen)
        # West road borders
        self.scene.addLine(0, center - R/2, center - R/2, center - R/2, glow_pen)
        self.scene.addLine(0, center + R/2, center - R/2, center + R/2, glow_pen)
        # East road borders
        self.scene.addLine(center + R/2, center - R/2, S, center - R/2, glow_pen)
        self.scene.addLine(center + R/2, center + R/2, S, center + R/2, glow_pen)

        # Lane divider (dashed center lines)
        marking_pen = QPen(QColor(COLORS["road_marking"]), 2, Qt.DashLine)
        # Vertical center line (above intersection)
        self.scene.addLine(center, 0, center, center - R/2, marking_pen)
        # Vertical center line (below intersection)
        self.scene.addLine(center, center + R/2, center, S, marking_pen)
        # Horizontal center line (left)
        self.scene.addLine(0, center, center - R/2, center, marking_pen)
        # Horizontal center line (right)
        self.scene.addLine(center + R/2, center, S, center, marking_pen)

        # Crosswalks (striped lines)
        cw_color = QColor(0, 255, 204, 35)  # Translucent neon turquoise
        cw_pen = QPen(Qt.NoPen)
        cw_w = 8
        stripe_gap = 10
        # North crosswalk
        for i in range(int(R / stripe_gap)):
            x = center - R/2 + i * stripe_gap
            self.scene.addRect(x, center - R/2 - 12, cw_w, 10, cw_pen, QBrush(cw_color))
        # South crosswalk
        for i in range(int(R / stripe_gap)):
            x = center - R/2 + i * stripe_gap
            self.scene.addRect(x, center + R/2 + 2, cw_w, 10, cw_pen, QBrush(cw_color))
        # West crosswalk
        for i in range(int(R / stripe_gap)):
            y = center - R/2 + i * stripe_gap
            self.scene.addRect(center - R/2 - 12, y, 10, cw_w, cw_pen, QBrush(cw_color))
        # East crosswalk
        for i in range(int(R / stripe_gap)):
            y = center - R/2 + i * stripe_gap
            self.scene.addRect(center + R/2 + 2, y, 10, cw_w, cw_pen, QBrush(cw_color))

        # Traffic lights (3-circle vertical/horizontal signal boxes)
        self._draw_traffic_lights(center, R)

        # Direction labels
        label_font = QFont("Consolas", 11, QFont.Bold)
        label_color = QColor(COLORS["accent_secondary"])
        for text, x, y in [("N", center-6, 8), ("S", center-4, S-24),
                            ("W", 8, center+4), ("E", S-20, center+4)]:
            t = self.scene.addText(text, label_font)
            t.setDefaultTextColor(label_color)
            t.setPos(x, y)

    def _draw_traffic_lights(self, center, R):
        """Draw traffic signal indicators at the intersection."""
        light_r = self.LIGHT_RADIUS
        box_pad = 4

        positions = {
            "north": (center + R/2 + 8, center - R/2 + 8),
            "south": (center - R/2 - 28, center + R/2 - 38),
            "east":  (center + R/2 - 38, center + R/2 + 8),
            "west":  (center - R/2 + 8, center - R/2 - 28),
        }

        for direction, (bx, by) in positions.items():
            # Signal box background
            is_vertical = direction in ("north", "south")
            if is_vertical:
                box_w, box_h = light_r * 2 + box_pad * 2, (light_r * 2 + box_pad) * 3 + box_pad
            else:
                box_w, box_h = (light_r * 2 + box_pad) * 3 + box_pad, light_r * 2 + box_pad * 2

            # Dark tech panel for signal housing with glowing border
            self.scene.addRect(bx, by, box_w, box_h, QPen(QColor(COLORS["border"]), 1.5), QBrush(QColor("#080911")))

            lights = []
            colors_order = ["signal_red", "signal_yellow", "signal_green"]
            for i, c_key in enumerate(colors_order):
                if is_vertical:
                    lx = bx + box_pad
                    ly = by + box_pad + i * (light_r * 2 + box_pad)
                else:
                    lx = bx + box_pad + i * (light_r * 2 + box_pad)
                    ly = by + box_pad

                dim_color = QColor(COLORS[c_key])
                dim_color.setAlpha(40)
                ellipse = self.scene.addEllipse(lx, ly, light_r*2, light_r*2,
                                                 QPen(Qt.NoPen), QBrush(dim_color))
                lights.append(ellipse)

            self._lights[direction] = lights  # [red, yellow, green]

    def update_state(self, intersection: Intersection, phase: SignalPhase):
        """Update vehicle positions and signal lights based on current state."""
        self._current_phase = phase

        # Update traffic lights
        self._update_lights(phase)

        # Update vehicles
        self._update_vehicles(intersection)

    def _update_lights(self, phase: SignalPhase):
        """Update traffic light colors based on current phase with radial gradient glow."""
        ns_green = phase == SignalPhase.NORTH_SOUTH_GREEN
        ew_green = phase == SignalPhase.EAST_WEST_GREEN
        is_yellow = phase == SignalPhase.YELLOW
        is_all_red = phase in (SignalPhase.ALL_RED, SignalPhase.PEDESTRIAN)

        light_r = self.LIGHT_RADIUS

        for direction, light_items in self._lights.items():
            red_item, yellow_item, green_item = light_items

            is_ns = direction in ("north", "south")
            is_this_green = (is_ns and ns_green) or (not is_ns and ew_green)

            # Reset all to dim solid color first
            for item, key in zip(light_items, ["signal_red", "signal_yellow", "signal_green"]):
                dim = QColor(COLORS[key])
                dim.setAlpha(40)
                item.setBrush(QBrush(dim))
                item.setPen(QPen(Qt.NoPen))

            # Apply glowing radial gradient to the active light
            active_item = None
            active_key = None
            if is_all_red:
                active_item = red_item
                active_key = "signal_red"
            elif is_yellow:
                active_item = yellow_item
                active_key = "signal_yellow"
            elif is_this_green:
                active_item = green_item
                active_key = "signal_green"
            else:
                active_item = red_item
                active_key = "signal_red"

            if active_item and active_key:
                rect = active_item.rect()
                cx = rect.x() + light_r
                cy = rect.y() + light_r
                grad = QRadialGradient(cx, cy, light_r + 4)
                grad.setColorAt(0.0, QColor("#ffffff"))  # White core
                grad.setColorAt(0.2, QColor(COLORS[active_key]))
                grad.setColorAt(0.7, QColor(COLORS[active_key]))
                # Soft glow edge
                fade = QColor(COLORS[active_key])
                fade.setAlpha(0)
                grad.setColorAt(1.0, fade)
                active_item.setBrush(QBrush(grad))
                # Add thin neon border
                active_item.setPen(QPen(QColor(COLORS[active_key]), 1.5))

    def _update_vehicles(self, intersection: Intersection):
        """Render vehicles as oriented rounded capsules in queue positions."""
        from PyQt5.QtWidgets import QGraphicsPathItem
        S = self.CANVAS_SIZE
        center = S / 2
        R = self.ROAD_WIDTH
        gap = self.VEHICLE_GAP
        max_visible = 14  # max vehicles to show per lane

        # Track which vehicle IDs are still active
        active_ids = set()

        for direction in Intersection.DIRECTIONS:
            lane = intersection.lanes[direction]
            vehicles = list(lane.queue)[:max_visible]

            for idx, v in enumerate(vehicles):
                active_ids.add(v.id)

                # Determine dimensions and headlight offset based on direction
                is_vertical = direction in ("north", "south")
                vw = 10 if is_vertical else 18
                vh = 18 if is_vertical else 10

                # Calculate position based on direction and queue index
                if direction == "north":
                    x = center + R/4 - vw/2  # right lane (approaching south)
                    y = center - R/2 - gap - idx * gap
                    hx, hy = vw/2, vh - 3    # headlight at bottom center
                elif direction == "south":
                    x = center - R/4 - vw/2  # left lane (approaching north)
                    y = center + R/2 + gap + idx * gap - vh
                    hx, hy = vw/2, 3         # headlight at top center
                elif direction == "east":
                    x = center + R/2 + gap + idx * gap - vw
                    y = center + R/4 - vh/2  # bottom lane (approaching west)
                    hx, hy = 3, vh/2         # headlight at left center
                elif direction == "west":
                    x = center - R/2 - gap - idx * gap
                    y = center - R/4 - vh/2  # top lane (approaching east)
                    hx, hy = vw - 3, vh/2    # headlight at right center

                # Bounds check
                if x < -vw or x > S + vw or y < -vh or y > S + vh:
                    continue

                color = QColor(COLORS["vehicle_emergency"]) if v.is_emergency else QColor(COLORS["vehicle_normal"])

                if v.id in self._vehicles:
                    # Move existing vehicle capsule
                    item = self._vehicles[v.id]
                    item.setPos(x, y)
                    item.setBrush(QBrush(color))
                else:
                    # Create new vehicle capsule path
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(0, 0, vw, vh), 3, 3)

                    item = QGraphicsPathItem(path)
                    item.setPen(QPen(QColor(COLORS["bg_dark"]), 1))
                    item.setBrush(QBrush(color))
                    item.setZValue(10)
                    item.setPos(x, y)

                    # Add headlight dot (neon cyan)
                    dot = QGraphicsEllipseItem(hx - 1.5, hy - 1.5, 3, 3, item)
                    dot.setBrush(QBrush(QColor(COLORS["accent_primary"])))
                    dot.setPen(QPen(Qt.NoPen))

                    self.scene.addItem(item)
                    self._vehicles[v.id] = item

        # Remove cleared vehicles
        to_remove = [vid for vid in self._vehicles if vid not in active_ids]
        for vid in to_remove:
            self.scene.removeItem(self._vehicles[vid])
            del self._vehicles[vid]

    def clear_vehicles(self):
        """Remove all vehicle graphics."""
        for item in self._vehicles.values():
            self.scene.removeItem(item)
        self._vehicles.clear()

    def reset_view(self):
        """Reset to initial state."""
        self._draw_intersection()


# ────────────────────────────────────────────────────────────
# Matplotlib Charts Panel
# ────────────────────────────────────────────────────────────

class ChartsPanel(QWidget):
    """Embedded matplotlib charts: line chart for wait time, bar chart for throughput, and baseline compare controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        
        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(10, 10, 10, 10)
        main_vbox.setSpacing(10)

        # Horizontal row for charts
        charts_row = QHBoxLayout()
        charts_row.setSpacing(10)

        # Wait Time Line Chart
        self.wait_fig = Figure(figsize=(5, 2.5), dpi=100)
        self.wait_fig.patch.set_facecolor(COLORS["bg_panel"])
        self.wait_ax = self.wait_fig.add_subplot(111)
        self.wait_canvas = FigureCanvas(self.wait_fig)
        self.wait_canvas.setStyleSheet(f"background-color: {COLORS['bg_panel']};")

        # Throughput Bar Chart
        self.throughput_fig = Figure(figsize=(5, 2.5), dpi=100)
        self.throughput_fig.patch.set_facecolor(COLORS["bg_panel"])
        self.throughput_ax = self.throughput_fig.add_subplot(111)
        self.throughput_canvas = FigureCanvas(self.throughput_fig)
        self.throughput_canvas.setStyleSheet(f"background-color: {COLORS['bg_panel']};")

        charts_row.addWidget(self.wait_canvas)
        charts_row.addWidget(self.throughput_canvas)
        main_vbox.addLayout(charts_row)

        # Bottom row for Compare to Baseline button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        
        self.compare_btn = QPushButton("  Compare with Baseline Controller")
        self.compare_btn.setEnabled(False)
        self.compare_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_secondary']};
                color: white;
                font-weight: bold;
                border: none;
                padding: 10px 24px;
                border-radius: 6px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: #6d28d9;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_input']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
            }}
        """)
        btn_row.addWidget(self.compare_btn)
        btn_row.addStretch()
        main_vbox.addLayout(btn_row)

        self._style_axes()

    def _style_axes(self):
        """Apply dark theme to matplotlib axes."""
        for ax in [self.wait_ax, self.throughput_ax]:
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
            ax.spines['bottom'].set_color(COLORS["border"])
            ax.spines['left'].set_color(COLORS["border"])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.label.set_color(COLORS["text_secondary"])
            ax.yaxis.label.set_color(COLORS["text_secondary"])
            ax.title.set_color(COLORS["accent_primary"])
            ax.title.set_fontsize(11)
            ax.grid(True, color=COLORS["border"], linestyle=":", linewidth=0.5, alpha=0.4)

        self.wait_ax.set_title("Average Wait Time")
        self.wait_ax.set_xlabel("Tick")
        self.wait_ax.set_ylabel("Wait (ticks)")

        self.throughput_ax.set_title("Vehicles Cleared")
        self.throughput_ax.set_xlabel("Direction")
        self.throughput_ax.set_ylabel("Count")

        self.wait_fig.tight_layout(pad=1.5)
        self.throughput_fig.tight_layout(pad=1.5)

    def update_charts(self, metrics: MetricsCollector):
        """Refresh both charts with current metrics data."""
        # Line chart
        self.wait_ax.clear()
        self._style_axes()
        self.wait_ax.set_title("Average Wait Time")
        self.wait_ax.set_xlabel("Tick")
        self.wait_ax.set_ylabel("Wait (ticks)")

        if metrics.avg_wait_history:
            x = list(range(len(metrics.avg_wait_history)))
            self.wait_ax.plot(x, metrics.avg_wait_history,
                            color=COLORS["accent_primary"], linewidth=1.5, alpha=0.9)
            self.wait_ax.fill_between(x, metrics.avg_wait_history,
                                      alpha=0.15, color=COLORS["accent_primary"])

        self.wait_fig.tight_layout(pad=1.5)
        self.wait_canvas.draw_idle()

        # Bar chart
        self.throughput_ax.clear()
        self._style_axes()
        self.throughput_ax.set_title("Vehicles Cleared")
        self.throughput_ax.set_xlabel("Direction")
        self.throughput_ax.set_ylabel("Count")

        dirs = ["North", "South", "East", "West"]
        vals = [
            metrics.throughput_per_direction.get("north", 0),
            metrics.throughput_per_direction.get("south", 0),
            metrics.throughput_per_direction.get("east", 0),
            metrics.throughput_per_direction.get("west", 0),
        ]
        bar_colors = [COLORS["accent_primary"], COLORS["accent_secondary"],
                      COLORS["accent_green"], COLORS["accent_orange"]]
        self.throughput_ax.bar(dirs, vals, color=bar_colors, edgecolor="none", width=0.6)

        self.throughput_fig.tight_layout(pad=1.5)
        self.throughput_canvas.draw_idle()

    def reset_charts(self):
        """Clear both charts."""
        self.wait_ax.clear()
        self.throughput_ax.clear()
        self._style_axes()
        self.wait_fig.tight_layout(pad=1.5)
        self.throughput_fig.tight_layout(pad=1.5)
        self.wait_canvas.draw_idle()
        self.throughput_canvas.draw_idle()


# ────────────────────────────────────────────────────────────
# Controls Panel
# ────────────────────────────────────────────────────────────

class ControlsPanel(QWidget):
    """Left sidebar with all simulation controls."""

    # Signals
    run_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    scenario_changed = pyqtSignal(str)
    duration_changed = pyqtSignal(int)
    emergency_toggled = pyqtSignal(bool)
    speed_changed = pyqtSignal(int)
    load_csv_clicked = pyqtSignal()
    info_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Scenario Selection ──
        scenario_group = QGroupBox("Scenario")
        sg_layout = QVBoxLayout(scenario_group)

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(["Light", "Moderate", "Heavy"])
        self.scenario_combo.currentTextChanged.connect(self._on_scenario_changed)
        sg_layout.addWidget(self.scenario_combo)

        # Top row buttons for CSV loading & Problem Info
        top_btn_layout = QHBoxLayout()
        self.load_csv_btn = QPushButton("Load CSV")
        self.load_csv_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; border-radius: 6px; padding: 5px 8px; font-size: 11px; }}
            QPushButton:hover {{ background-color: {COLORS['accent_primary']}; color: {COLORS['bg_dark']}; }}
        """)
        self.load_csv_btn.clicked.connect(self.load_csv_clicked.emit)

        self.info_btn = QPushButton("Setup Info")
        self.info_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS['bg_card']}; border: 1px solid {COLORS['border']}; border-radius: 6px; padding: 5px 8px; font-size: 11px; }}
            QPushButton:hover {{ background-color: {COLORS['accent_primary']}; color: {COLORS['bg_dark']}; }}
        """)
        self.info_btn.clicked.connect(self.info_clicked.emit)

        top_btn_layout.addWidget(self.load_csv_btn)
        top_btn_layout.addWidget(self.info_btn)
        sg_layout.addLayout(top_btn_layout)

        layout.addWidget(scenario_group)

        # ── Duration ──
        duration_group = QGroupBox("Duration (ticks)")
        dg_layout = QHBoxLayout(duration_group)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(50, 1000)
        self.duration_spin.setValue(300)
        self.duration_spin.setSingleStep(50)
        self.duration_spin.valueChanged.connect(lambda v: self.duration_changed.emit(v))
        dg_layout.addWidget(self.duration_spin)
        layout.addWidget(duration_group)

        # ── Emergency Mode ──
        self.emergency_check = QCheckBox("  Emergency Vehicle Mode")
        self.emergency_check.setChecked(True)
        self.emergency_check.stateChanged.connect(
            lambda s: self.emergency_toggled.emit(s == Qt.Checked))
        self.emergency_check.setStyleSheet(f"""
            QCheckBox {{ padding: 8px; background-color: {COLORS['bg_panel']};
                        border: 1px solid {COLORS['border']}; border-radius: 8px; }}
        """)
        layout.addWidget(self.emergency_check)

        # ── Control Buttons ──
        btn_group = QGroupBox("Controls")
        bg_layout = QVBoxLayout(btn_group)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("  Run")
        self.run_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS['accent_green']}; color: white;
                          font-weight: bold; border: none; }}
            QPushButton:hover {{ background-color: #0ea572; }}
        """)
        self.run_btn.clicked.connect(self.run_clicked.emit)

        self.pause_btn = QPushButton("  Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS['accent_yellow']}; color: #1a1a2e;
                          font-weight: bold; border: none; }}
            QPushButton:hover {{ background-color: #d97706; }}
        """)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)

        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.pause_btn)
        bg_layout.addLayout(btn_row)

        self.reset_btn = QPushButton("  Reset")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        bg_layout.addWidget(self.reset_btn)

        layout.addWidget(btn_group)

        # ── Speed Control ──
        speed_group = QGroupBox("Speed")
        spg_layout = QHBoxLayout(speed_group)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 20)
        self.speed_slider.setValue(5)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(5)

        self.speed_label = QLabel("5x")
        self.speed_label.setFixedWidth(30)
        self.speed_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-weight: bold;")
        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        spg_layout.addWidget(self.speed_slider)
        spg_layout.addWidget(self.speed_label)
        layout.addWidget(speed_group)

        layout.addStretch()

        # ── KPI Display ──
        kpi_group = QGroupBox("Live KPIs")
        kg_layout = QVBoxLayout(kpi_group)

        self.kpi_wait = QLabel("Avg Wait: 0.0s")
        self.kpi_max_wait = QLabel("Max Wait: 0.0s")
        self.kpi_emergency_wait = QLabel("Emergency Wait: 0.0s")
        self.kpi_throughput = QLabel("Throughput: 0")
        self.kpi_runtime = QLabel("Tick Time: 0.000ms")
        for lbl in [self.kpi_wait, self.kpi_max_wait, self.kpi_emergency_wait, self.kpi_throughput, self.kpi_runtime]:
            lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; padding: 2px;")
            kg_layout.addWidget(lbl)

        layout.addWidget(kpi_group)

    def _on_scenario_changed(self, text):
        self.scenario_changed.emit(text)

    def _on_speed_changed(self, value):
        self.speed_label.setText(f"{value}x")
        self.speed_changed.emit(value)

    def update_kpis(self, avg_wait: float, throughput: int, runtime_ms: float, max_wait: float = 0.0, emergency_wait: float = 0.0):
        self.kpi_wait.setText(f"Avg Wait: {avg_wait:.1f}s")
        self.kpi_max_wait.setText(f"Max Wait: {max_wait:.1f}s")
        self.kpi_emergency_wait.setText(f"Emergency Wait: {emergency_wait:.1f}s")
        self.kpi_throughput.setText(f"Throughput: {throughput}")
        self.kpi_runtime.setText(f"Tick Time: {runtime_ms:.3f}ms")

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.scenario_combo.setEnabled(not running)
        self.load_csv_btn.setEnabled(not running)
        self.duration_spin.setEnabled(not running)

    def set_completed(self):
        self.run_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)


# ────────────────────────────────────────────────────────────
# Explanation Panel
# ────────────────────────────────────────────────────────────

class ExplanationPanel(QWidget):
    """Right sidebar: AI explanations, factor breakdown, chat."""

    chat_submitted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)

        # Wrap everything in a scroll area so nothing gets cut off
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background-color: transparent; }}")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Current Decision ──
        decision_group = QGroupBox("Current Decision")
        dg_layout = QVBoxLayout(decision_group)

        self.decision_label = QLabel("Waiting for simulation...")
        self.decision_label.setWordWrap(True)
        self.decision_label.setStyleSheet(f"color: {COLORS['accent_primary']}; font-size: 13px; padding: 4px;")
        dg_layout.addWidget(self.decision_label)

        layout.addWidget(decision_group)

        # ── Factor Breakdown Table ──
        factors_group = QGroupBox("Decision Factors")
        fg_layout = QVBoxLayout(factors_group)

        self.factors_table = QTableWidget(4, 4)
        self.factors_table.setHorizontalHeaderLabels(["Lane", "Queue", "Wait", "Emg"])
        self.factors_table.verticalHeader().setVisible(False)
        self.factors_table.setMinimumHeight(170)
        self.factors_table.setMaximumHeight(200)
        self.factors_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.factors_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set explicit row heights so all 4 rows are fully visible
        for row_idx in range(4):
            self.factors_table.setRowHeight(row_idx, 32)

        # Populate with defaults
        for i, d in enumerate(["North", "South", "East", "West"]):
            self.factors_table.setItem(i, 0, QTableWidgetItem(d))
            self.factors_table.setItem(i, 1, QTableWidgetItem("0"))
            self.factors_table.setItem(i, 2, QTableWidgetItem("0.0"))
            self.factors_table.setItem(i, 3, QTableWidgetItem("-"))

        fg_layout.addWidget(self.factors_table)
        layout.addWidget(factors_group)

        # ── AI Explanation History ──
        ai_group = QGroupBox("AI Explanation")
        ag_layout = QVBoxLayout(ai_group)

        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMinimumHeight(140)
        self.explanation_text.setMaximumHeight(260)
        self.explanation_text.setPlaceholderText("AI explanations will appear here...")
        ag_layout.addWidget(self.explanation_text)

        layout.addWidget(ai_group)

        # ── Chat ──
        chat_group = QGroupBox("Ask AI")
        cg_layout = QVBoxLayout(chat_group)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(100)
        self.chat_display.setMaximumHeight(200)
        self.chat_display.setPlaceholderText("Chat with the AI about decisions...")
        cg_layout.addWidget(self.chat_display)

        chat_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a question...")
        self.chat_input.returnPressed.connect(self._on_chat_submit)

        self.chat_send_btn = QPushButton("Send")
        self.chat_send_btn.setFixedWidth(60)
        self.chat_send_btn.setStyleSheet(f"""
            QPushButton {{ background-color: {COLORS['accent_primary']}; color: {COLORS['bg_dark']};
                          font-weight: bold; border: none; }}
            QPushButton:hover {{ background-color: #00b8d4; }}
        """)
        self.chat_send_btn.clicked.connect(self._on_chat_submit)

        chat_row.addWidget(self.chat_input)
        chat_row.addWidget(self.chat_send_btn)
        cg_layout.addLayout(chat_row)

        layout.addWidget(chat_group)
        layout.addStretch()

        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

    def _on_chat_submit(self):
        text = self.chat_input.text().strip()
        if text:
            self.chat_input.clear()
            self.chat_submitted.emit(text)

    def update_decision(self, decision: PhaseDecision):
        """Update the decision display and factor table."""
        phase_name = decision.chosen_phase.name.replace("_", " ").title()
        self.decision_label.setText(f"Tick {decision.tick}: {phase_name}")

        dirs = ["north", "south", "east", "west"]
        for i, d in enumerate(dirs):
            self.factors_table.setItem(i, 0, QTableWidgetItem(d.capitalize()))
            q = decision.queue_lengths.get(d, 0)
            w = decision.avg_wait_times.get(d, 0)
            e = decision.emergency_flags.get(d, False)

            q_item = QTableWidgetItem(str(q))
            w_item = QTableWidgetItem(f"{w:.1f}")
            e_item = QTableWidgetItem("!!" if e else "-")

            if e:
                e_item.setForeground(QColor(COLORS["accent_red"]))
                e_item.setFont(QFont("Segoe UI", 11, QFont.Bold))

            self.factors_table.setItem(i, 1, q_item)
            self.factors_table.setItem(i, 2, w_item)
            self.factors_table.setItem(i, 3, e_item)

    def add_explanation(self, tick: int, text: str):
        """Add an AI explanation to the history."""
        formatted_text = self.markdown_to_html(text)
        html = f'<p style="margin:4px 0;"><b style="color:{COLORS["accent_primary"]}">Tick {tick}:</b> {formatted_text}</p>'
        self.explanation_text.append(html)

    def add_chat_message(self, sender: str, text: str):
        """Add a chat message to the display."""
        if sender == "user":
            color = COLORS["accent_primary"]
            prefix = "You"
        else:
            color = COLORS["accent_green"]
            prefix = "AI"
        formatted_text = self.markdown_to_html(text)
        html = f'<p style="margin:4px 0;"><b style="color:{color}">{prefix}:</b> {formatted_text}</p>'
        self.chat_display.append(html)

    def markdown_to_html(self, text: str) -> str:
        """Convert basic Markdown elements to HTML for rendering in QTextEdit."""
        # Convert headers like "### Title" to colored bold text
        text = re.sub(r'(?m)^#{1,6}\s+(.*)$', r'<b style="color:' + COLORS["accent_primary"] + r'; font-size:14px;">\1</b>', text)
        # Convert markdown bullet points to bullet characters
        text = re.sub(r'(?m)^[\s]*[\*\-][\s]+(.*)$', r'• \1', text)
        # Convert bold **text** to <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Convert italic *text* to <i>text</i>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        # Convert newlines to HTML breaks to preserve spacing
        text = text.replace('\n', '<br>')
        return text

    def reset(self):
        self.decision_label.setText("Waiting for simulation...")
        for i in range(4):
            self.factors_table.setItem(i, 1, QTableWidgetItem("0"))
            self.factors_table.setItem(i, 2, QTableWidgetItem("0.0"))
            self.factors_table.setItem(i, 3, QTableWidgetItem("-"))
        self.explanation_text.clear()
        self.chat_display.clear()


# ────────────────────────────────────────────────────────────
# Comparison Dialog
# ────────────────────────────────────────────────────────────

class ComparisonDialog(QDialog):
    """Side-by-side comparison of AI optimizer vs fixed-timer baseline."""

    def __init__(self, ai_summary: dict, baseline_summary: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI vs Baseline Comparison")
        self.setMinimumSize(850, 620)
        self.setStyleSheet(DARK_STYLESHEET)

        layout = QVBoxLayout(self)

        title = QLabel("Performance Comparison: AI Optimizer vs Fixed Timer")
        title.setStyleSheet(f"color: {COLORS['accent_primary']}; font-size: 16px; font-weight: bold; padding: 10px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Chart (Grid layout for 5 metrics)
        fig = Figure(figsize=(9, 5), dpi=100)
        fig.patch.set_facecolor(COLORS["bg_panel"])
        canvas = FigureCanvas(fig)

        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)

        for ax in [ax1, ax2, ax3, ax4, ax5]:
            ax.set_facecolor(COLORS["bg_card"])
            ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
            ax.spines['bottom'].set_color(COLORS["border"])
            ax.spines['left'].set_color(COLORS["border"])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.title.set_color(COLORS["accent_primary"])
            ax.title.set_fontsize(10)
            ax.grid(True, color=COLORS["border"], linestyle=":", linewidth=0.5, alpha=0.4)

        # 1) Avg Wait Time
        vals = [ai_summary.get("avg_wait_time", 0.0), baseline_summary.get("avg_wait_time", 0.0)]
        bars = ax1.bar(["AI", "Baseline"], vals,
                       color=[COLORS["accent_primary"], COLORS["accent_red"]],
                       edgecolor="none", width=0.5)
        ax1.set_title("Avg Wait (ticks)")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", color=COLORS["text_primary"], fontsize=9)

        # 2) Max Wait Time
        vals = [ai_summary.get("max_wait_time", 0.0), baseline_summary.get("max_wait_time", 0.0)]
        bars = ax2.bar(["AI", "Baseline"], vals,
                       color=[COLORS["accent_primary"], COLORS["accent_red"]],
                       edgecolor="none", width=0.5)
        ax2.set_title("Max Wait (ticks)")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", color=COLORS["text_primary"], fontsize=9)

        # 3) Total Throughput
        vals = [ai_summary.get("total_throughput", 0), baseline_summary.get("total_throughput", 0)]
        bars = ax3.bar(["AI", "Baseline"], vals,
                       color=[COLORS["accent_primary"], COLORS["accent_red"]],
                       edgecolor="none", width=0.5)
        ax3.set_title("Total Throughput")
        for bar, val in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha="center", color=COLORS["text_primary"], fontsize=9)

        # 4) Emergency Wait Time
        vals = [ai_summary.get("emergency_wait_time", 0.0), baseline_summary.get("emergency_wait_time", 0.0)]
        bars = ax4.bar(["AI", "Baseline"], vals,
                       color=[COLORS["accent_primary"], COLORS["accent_red"]],
                       edgecolor="none", width=0.5)
        ax4.set_title("Emg Wait (ticks)")
        for bar, val in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", color=COLORS["text_primary"], fontsize=9)

        # 5) Runtime
        vals = [ai_summary.get("avg_tick_runtime_ms", 0.0), baseline_summary.get("avg_tick_runtime_ms", 0.0)]
        bars = ax5.bar(["AI", "Baseline"], vals,
                       color=[COLORS["accent_primary"], COLORS["accent_red"]],
                       edgecolor="none", width=0.5)
        ax5.set_title("Runtime (ms)")
        for bar, val in zip(bars, vals):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", color=COLORS["text_primary"], fontsize=9)

        fig.tight_layout(pad=1.5)
        layout.addWidget(canvas)

        # Summary table
        summary_group = QGroupBox("Performance Summary Table")
        sg_layout = QVBoxLayout(summary_group)

        table = QTableWidget(5, 3)
        table.setHorizontalHeaderLabels(["Metric / KPI", "AI Optimizer", "Fixed Timer"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setMaximumHeight(180)

        metrics_data = [
            ("Avg Wait Time", f"{ai_summary.get('avg_wait_time', 0.0):.1f} ticks",
             f"{baseline_summary.get('avg_wait_time', 0.0):.1f} ticks"),
            ("Max Wait Time", f"{ai_summary.get('max_wait_time', 0.0):.1f} ticks",
             f"{baseline_summary.get('max_wait_time', 0.0):.1f} ticks"),
            ("Emergency Wait Time", f"{ai_summary.get('emergency_wait_time', 0.0):.1f} ticks",
             f"{baseline_summary.get('emergency_wait_time', 0.0):.1f} ticks"),
            ("Total Throughput", str(ai_summary.get("total_throughput", 0)),
             str(baseline_summary.get("total_throughput", 0))),
            ("Avg Tick Runtime", f"{ai_summary.get('avg_tick_runtime_ms', 0.0):.3f} ms",
             f"{baseline_summary.get('avg_tick_runtime_ms', 0.0):.3f} ms"),
        ]

        for i, (name, ai_val, bl_val) in enumerate(metrics_data):
            table.setItem(i, 0, QTableWidgetItem(name))
            ai_item = QTableWidgetItem(ai_val)
            bl_item = QTableWidgetItem(bl_val)
            ai_item.setForeground(QColor(COLORS["accent_primary"]))
            bl_item.setForeground(QColor(COLORS["accent_red"]))
            table.setItem(i, 1, ai_item)
            table.setItem(i, 2, bl_item)

        sg_layout.addWidget(table)
        layout.addWidget(summary_group)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(120)
        close_btn.clicked.connect(self.accept)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)


# ────────────────────────────────────────────────────────────
# Main Window
# ────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window — wires all components together."""

    explanation_ready = pyqtSignal(int, str)
    chat_ready = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Traffic Signal Controller")
        self.setMinimumSize(1200, 780)
        self.resize(1400, 850)
        self.setStyleSheet(DARK_STYLESHEET)

        # ── Load presets ──
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        presets_path = os.path.join(data_dir, "presets.json")
        self.presets = load_preset_scenarios(presets_path)
        self.constraints = SimConstraints.from_presets_file(presets_path)

        # ── AI Explainer ──
        # Load API key from global constant or environment variable
        api_key = API_KEY if API_KEY != "your-api-key-here" else os.environ.get("GEMINI_API_KEY", "")
        self.explainer = GeminiExplainer(api_key=api_key)

        # ── State ──
        self.current_rates = {"north": 8, "south": 7, "east": 6, "west": 9}
        self._scenario_data = []
        self._runner: Optional[SimulationRunner] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._speed = 5  # ticks per second
        self._is_paused = False
        self._ai_summary = None
        self._last_api_call_time = 0.0

        # ── Build UI ──
        self._build_ui()
        self._connect_signals()

        # ── Load default scenario ──
        self._load_scenario("Moderate")

        # ── Status bar ──
        self.statusBar().showMessage("Ready -- Select a scenario and click Run")

    def _build_ui(self):
        """Construct the main window layout with a persistent sidebar and a 3-tab diagnostic view."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # Left: persistent control sidebar
        self.controls = ControlsPanel()
        main_layout.addWidget(self.controls)

        # Right: Tab Widget containing the three core views
        self.tabs = QTabWidget()
        
        # Tab 1: Intersection Simulation
        sim_tab_widget = QWidget()
        sim_tab_layout = QVBoxLayout(sim_tab_widget)
        sim_tab_layout.setContentsMargins(10, 10, 10, 10)
        sim_tab_layout.setSpacing(8)

        # Phase indicator above intersection
        self.phase_bar = QLabel("Phase: NORTH-SOUTH GREEN | Timer: --")
        self.phase_bar.setAlignment(Qt.AlignCenter)
        self.phase_bar.setMaximumHeight(35)
        self.phase_bar.setStyleSheet(f"""
            background-color: {COLORS['bg_panel']};
            color: {COLORS['accent_primary']};
            font-size: 12px;
            font-weight: bold;
            padding: 4px;
            border-radius: 6px;
            border: 1px solid {COLORS['border']};
        """)

        sim_tab_layout.addWidget(self.phase_bar)

        self.intersection_view = IntersectionCanvas()
        sim_tab_layout.addWidget(self.intersection_view)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 300)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Tick %v / %m")
        sim_tab_layout.addWidget(self.progress)

        self.tabs.addTab(sim_tab_widget, "Intersection Simulation")

        # Tab 2: AI Decisions & Chat Explanations
        self.explanation = ExplanationPanel()
        self.tabs.addTab(self.explanation, "AI Explanations & Logs")

        # Tab 3: Performance Charts
        self.charts = ChartsPanel()
        self.tabs.addTab(self.charts, "Real-Time Charts")

        main_layout.addWidget(self.tabs, stretch=3)

    def _connect_signals(self):
        """Wire up all UI signals to handlers."""
        self.controls.run_clicked.connect(self._on_run)
        self.controls.pause_clicked.connect(self._on_pause)
        self.controls.reset_clicked.connect(self._on_reset)
        self.controls.scenario_changed.connect(self._on_scenario_changed)
        self.controls.duration_changed.connect(self._on_duration_changed)
        self.controls.emergency_toggled.connect(self._on_emergency_toggled)
        self.controls.speed_changed.connect(self._on_speed_changed)
        self.charts.compare_btn.clicked.connect(self._on_compare)
        self.controls.load_csv_clicked.connect(self._on_load_csv)
        self.controls.info_clicked.connect(self._on_show_info)
        self.explanation.chat_submitted.connect(self._on_chat)
        self.explanation_ready.connect(self.explanation.add_explanation)
        self.chat_ready.connect(lambda text: self.explanation.add_chat_message("ai", text))

    # ── Scenario management ──

    def _load_scenario(self, name: str):
        """Load a preset scenario or prepare for custom."""
        if name in self.presets:
            preset = self.presets[name]

            # Remove "Custom" item if switching back to preset scenario
            self.controls.scenario_combo.blockSignals(True)
            custom_idx = self.controls.scenario_combo.findText("Custom")
            if custom_idx != -1:
                self.controls.scenario_combo.removeItem(custom_idx)
            self.controls.scenario_combo.blockSignals(False)

            rates = preset["arrival_rates"]
            duration = preset["duration_ticks"]
            emg_prob = preset["emergency_probability"]

            self.current_rates = rates
            self.controls.duration_spin.setValue(duration)

            self._scenario_data = generate_scenario_data(rates, duration, emg_prob)
            self.progress.setRange(0, duration)
            self.progress.setValue(0)

            self._update_status("READY", f"Loaded '{name}' scenario: {preset['description']}")

    def _on_scenario_changed(self, text):
        self._load_scenario(text)

    def _on_duration_changed(self, val):
        self._scenario_data = generate_scenario_data(self.current_rates, val)
        self.progress.setRange(0, val)
        self._update_status("READY", f"Simulation duration updated to {val} ticks")

    def _on_emergency_toggled(self, enabled):
        if self._runner:
            self._runner.emergency_mode = enabled
            self._update_status("SIMULATING" if self._runner.is_running else "IDLE", 
                                f"Emergency mode {'enabled' if enabled else 'disabled'}")

    def _on_speed_changed(self, value):
        self._speed = value
        if self._timer.isActive():
            self._timer.setInterval(max(16, 1000 // self._speed))

    # ── Simulation control ──

    def _on_run(self):
        """Start or resume the simulation."""
        if self._is_paused and self._runner:
            self._is_paused = False
            self._timer.start(max(16, 1000 // self._speed))
            self.controls.set_running(True)
            self._update_status("SIMULATING", "Simulation resumed")
            return

        if not self._scenario_data:
            self._update_status("ERROR", "No scenario data loaded!")
            return

        # Create fresh runner
        self._runner = SimulationRunner(
            scenario_data=self._scenario_data,
            controller=PriorityOptimizer(self.constraints),
            constraints=self.constraints,
            emergency_mode=self.controls.emergency_check.isChecked(),
        )

        self.explainer.clear_history()
        self.controls.set_running(True)
        self.charts.compare_btn.setEnabled(False)
        self._is_paused = False

        self._timer.start(max(16, 1000 // self._speed))
        self._update_status("SIMULATING", "Simulation running...")

    def _on_pause(self):
        """Pause the simulation."""
        self._is_paused = True
        self._timer.stop()
        self.controls.run_btn.setEnabled(True)
        self.controls.pause_btn.setEnabled(False)
        self._update_status("IDLE", "Simulation paused")

    def _on_reset(self):
        """Reset everything to initial state."""
        self._timer.stop()
        self._runner = None
        self._is_paused = False
        self._ai_summary = None

        self.intersection_view.reset_view()
        self.charts.reset_charts()
        self.explanation.reset()
        self.controls.set_running(False)
        self.charts.compare_btn.setEnabled(False)
        self.progress.setValue(0)
        self.controls.update_kpis(0, 0, 0, 0, 0)
        self.phase_bar.setText("Phase: NORTH-SOUTH GREEN | Timer: --")

        # Reload scenario
        self._load_scenario(self.controls.scenario_combo.currentText())
        self._update_status("READY", "Simulation reset")

    # ── Timer tick (drives simulation) ──

    def _on_timer_tick(self):
        """Called by QTimer — advance simulation by one tick."""
        if not self._runner or self._runner.is_completed:
            self._timer.stop()
            self._on_simulation_complete()
            return

        decision = self._runner.step()

        # Update intersection view
        self.intersection_view.update_state(
            self._runner.intersection,
            self._runner.intersection.current_phase
        )

        # Update progress
        self.progress.setValue(self._runner.current_tick)

        # Update phase bar
        phase = self._runner.intersection.current_phase
        phase_name = phase.name.replace("_", " ").title()
        elapsed = self._runner._elapsed_in_phase
        self.phase_bar.setText(f"Phase: {phase_name} | Elapsed: {elapsed}s")

        # Update phase bar color
        if phase == SignalPhase.NORTH_SOUTH_GREEN or phase == SignalPhase.EAST_WEST_GREEN:
            bar_color = COLORS["accent_green"]
        elif phase == SignalPhase.YELLOW:
            bar_color = COLORS["accent_yellow"]
        else:
            bar_color = COLORS["accent_red"]

        self.phase_bar.setStyleSheet(f"""
            background-color: {COLORS['bg_panel']};
            color: {bar_color};
            font-size: 14px;
            font-weight: bold;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid {bar_color};
        """)

        # Update charts every 5 ticks
        if self._runner.current_tick % 5 == 0:
            self.charts.update_charts(self._runner.metrics)

        # Update KPIs
        m = self._runner.metrics
        self.controls.update_kpis(m.overall_avg_wait, m.total_cleared, m.avg_tick_runtime, m.overall_max_wait, m.overall_emergency_wait)

        # Handle decisions — update explanation panel
        if decision:
            self.explanation.update_decision(decision)
            context = self.explainer.build_decision_context(decision)

            def on_explanation(text, tick=decision.tick):
                self.explanation_ready.emit(tick, text)

            if self.explainer.is_available:
                now = time.time()
                if (now - self._last_api_call_time) >= 4.0:
                    self._last_api_call_time = now
                    self.explainer.explain_decision(context, callback=on_explanation)
                else:
                    # Cooldown period: show fallback immediately to keep user informed without delay
                    fallback_text = f"[API Cooldown Fallback] {self.explainer.fallback.explain_decision(context)}"
                    self.explanation_ready.emit(decision.tick, fallback_text)
            else:
                # Use fallback if the API is not set up / offline
                fallback_text = self.explainer.fallback.explain_decision(context)
                self.explanation_ready.emit(decision.tick, fallback_text)

        # Update status
        self.statusBar().showMessage(
            f"Running | Tick {self._runner.current_tick}/{self._runner.total_ticks} | "
            f"Phase: {phase_name}"
        )

    def _on_simulation_complete(self):
        """Handle simulation completion."""
        self.controls.set_completed()
        self._ai_summary = self._runner.metrics.get_summary() if self._runner else None

        # Final chart update
        if self._runner:
            self.charts.update_charts(self._runner.metrics)
            m = self._runner.metrics
            self.controls.update_kpis(m.overall_avg_wait, m.total_cleared, m.avg_tick_runtime, m.overall_max_wait, m.overall_emergency_wait)

        self._update_status("READY", "Simulation complete! Click 'Compare to Baseline' to see results.")

    # ── Compare to Baseline ──

    def _on_compare(self):
        """Run baseline and show comparison dialog."""
        if not self._ai_summary or not self._scenario_data:
            return

        self.statusBar().showMessage("Running baseline comparison...")
        QApplication.processEvents()

        # Run fixed-timer baseline
        baseline_runner = SimulationRunner(
            scenario_data=self._scenario_data,
            controller=FixedTimerController(self.constraints),
            constraints=self.constraints,
            emergency_mode=self.controls.emergency_check.isChecked(),
        )
        baseline_summary = baseline_runner.run_all()

        # Show dialog
        dialog = ComparisonDialog(self._ai_summary, baseline_summary, self)
        dialog.exec_()
        self.statusBar().showMessage("Comparison complete")

    # ── Chat ──

    def _on_chat(self, question: str):
        """Handle user chat question."""
        self.explanation.add_chat_message("user", question)

        if not self._runner:
            self.explanation.add_chat_message("ai", "Please run a simulation first to ask questions about decisions.")
            return

        sim_state = self.explainer.build_sim_state(
            self._runner.intersection,
            self._runner.current_tick,
            self._runner.decisions[-1] if self._runner.decisions else None
        )

        def on_answer(text):
            self.chat_ready.emit(text)

        self.explainer.answer_question(question, sim_state, callback=on_answer)

    # ── PDF Support Functions ──

    def _update_status(self, state: str, message: Optional[str] = None):
        """Update status bar message."""
        if message:
            self.statusBar().showMessage(message)

    def _on_show_info(self):
        """Displays the Problem Setup Info dialog detailing inputs, outputs, and constraints."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Problem Setup & System Constraints")
        dialog.setMinimumSize(500, 450)
        dialog.setStyleSheet(DARK_STYLESHEET)
        
        layout = QVBoxLayout(dialog)
        
        title = QLabel("Smart Traffic Signal Controller Info")
        title.setStyleSheet(f"color: {COLORS['accent_primary']}; font-size: 15px; font-weight: bold; padding: 5px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setStyleSheet(f"background-color: {COLORS['bg_card']}; border: 1px solid {COLORS['border']};")
        
        html_content = f"""
        <h3>Problem Setup & Definition</h3>
        <p><b>Goal:</b> Optimize traffic signal phases at a 4-way intersection dynamically to minimize vehicle wait times and prevent congestion, with a hard override for emergency vehicles.</p>
        
        <h4>1. Simulation Inputs</h4>
        <ul>
            <li><b>Vehicle Arrival Rates:</b> Configured per lane (North, South, East, West) from 1 to 25 vehicles/minute. Modeled as a Poisson arrival process.</li>
            <li><b>Emergency Vehicle Probability:</b> Chance of spawning emergency vehicles (e.g. ambulances/fire trucks) on any direction.</li>
            <li><b>Scenario Presets:</b> Light, Moderate, Heavy, or Custom parameters.</li>
            <li><b>Simulation Duration:</b> Total runtime of the simulation (in ticks).</li>
        </ul>
        
        <h4>2. Optimization Constraints</h4>
        <ul>
            <li><b>Minimum Green Time (8 ticks):</b> A green phase must run for at least 8 ticks before the system can switch directions, preventing rapid signal flickering.</li>
            <li><b>Yellow Transition Time (3 ticks):</b> Safe deceleration window when switching phases.</li>
            <li><b>All-Red Phase (1 tick):</b> Brief safety window where all signals are red to clear the intersection.</li>
            <li><b>Pedestrian Cycle (10 ticks):</b> A dedicated pedestrian crossing phase is triggered automatically every 4th full signal cycle.</li>
            <li><b>Emergency Override:</b> Signals switch immediately to favor an opposing direction if an emergency vehicle is detected there and not in the current direction.</li>
        </ul>
        
        <h4>3. System Outputs & KPIs</h4>
        <ul>
            <li><b>Average Wait Time:</b> Accumulated tick delay of vehicles waiting in the queues.</li>
            <li><b>Maximum Wait Time:</b> The maximum delay experienced by any single vehicle.</li>
            <li><b>Emergency Wait Time:</b> Average response delay for prioritized emergency vehicles.</li>
            <li><b>Throughput:</b> Total number of vehicles successfully cleared.</li>
            <li><b>Decision Explanations:</b> Natural language justifications generated by the Gemini AI Explainer.</li>
        </ul>
        """
        info_text.setHtml(html_content)
        layout.addWidget(info_text)
        
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(dialog.accept)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        dialog.exec_()

    def _on_load_csv(self):
        """Allows user to upload/load a custom scenario CSV and validates it."""
        from traffic_engine import load_data, preprocess_data
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Custom Scenario CSV", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return  # User canceled
            
        self._update_status("LOADING", "Loading custom CSV scenario...")
        
        try:
            # 1) Load data
            raw_data = load_data(file_path)
            
            # 2) Validate data
            validated_data = preprocess_data(raw_data)
            
            # 3) Store data & setup state
            self._scenario_data = validated_data
            duration = len(self._scenario_data)
            
            # Switch preset dropdown to "Custom" without triggering reload
            self.controls.scenario_combo.blockSignals(True)
            if self.controls.scenario_combo.findText("Custom") == -1:
                self.controls.scenario_combo.addItem("Custom")
            self.controls.scenario_combo.setCurrentText("Custom")
            self.controls.scenario_combo.blockSignals(False)
            
            self.controls.duration_spin.setValue(duration)
            self.progress.setRange(0, duration)
            self.progress.setValue(0)
            
            self._update_status("READY", f"Custom CSV loaded successfully ({duration} ticks)")
            QMessageBox.information(
                self, "CSV Loaded Successfully", 
                f"Successfully loaded and validated {duration} ticks of traffic data from:\n{os.path.basename(file_path)}"
            )
            
        except Exception as e:
            self._update_status("ERROR", f"Error loading CSV: {str(e)}")
            QMessageBox.critical(
                self, "Error Loading CSV", 
                f"The CSV file could not be loaded because it failed validation:\n\n{str(e)}"
            )


# ────────────────────────────────────────────────────────────
# PDF Suggested Wrapper Functions
# ────────────────────────────────────────────────────────────

def create_visuals(data, result):
    """Wrapper function to handle UI canvas and chart updates (Required by Lab Project Guide)."""
    # In this PyQt desktop application, visual updates are driven live by MainWindow timers
    # and Matplotlib charts. This wrapper provides compatibility with Section 4 of the PDF.
    pass


def render_ui():
    """Wrapper function to instantiate and run the desktop user interface (Required by Lab Project Guide)."""
    main()


# ────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set app-wide dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg_dark"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Base, QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["bg_panel"]))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Button, QColor(COLORS["bg_card"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent_primary"]))
    palette.setColor(QPalette.HighlightedText, QColor(COLORS["bg_dark"]))
    app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
