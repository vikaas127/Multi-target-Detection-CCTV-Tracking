import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QDate

LOG_FILE = "activity_log.csv"

class ReportDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Worker Activity Dashboard")
        self.setGeometry(100, 100, 1400, 800)

        # --- Layouts ---
        central = QWidget()
        self.setCentralWidget(central)
        hbox = QHBoxLayout(central)

        # Left Filters
        filter_panel = QVBoxLayout()
        hbox.addLayout(filter_panel, 1)

        # Date Range
        filter_panel.addWidget(QLabel("Date Range"))
        self.start_date = QDateEdit(calendarPopup=True)
        self.end_date = QDateEdit(calendarPopup=True)
        self.start_date.setDate(QDate.currentDate().addDays(-7))
        self.end_date.setDate(QDate.currentDate())
        filter_panel.addWidget(self.start_date)
        filter_panel.addWidget(self.end_date)

        # Worker Filter
        filter_panel.addWidget(QLabel("Worker"))
        self.worker_filter = QComboBox()
        filter_panel.addWidget(self.worker_filter)

        # Camera Filter
        filter_panel.addWidget(QLabel("Camera"))
        self.camera_filter = QComboBox()
        filter_panel.addWidget(self.camera_filter)

        # Apply Button
        self.apply_btn = QPushButton("Apply Filters")
        self.apply_btn.clicked.connect(self.load_data)
        filter_panel.addWidget(self.apply_btn)

        # Chart Area
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        hbox.addWidget(self.canvas, 3)

        # Load data first time
        self.load_data()

    def load_data(self):
        try:
            df = pd.read_csv(LOG_FILE)
        except:
            QMessageBox.warning(self, "Error", "No activity log found!")
            return

        # Convert Date column to datetime
        df["Date"] = pd.to_datetime(df["Date"])

        # Fill filters dynamically
        self.worker_filter.clear()
        self.worker_filter.addItem("All")
        self.worker_filter.addItems(sorted(df["Name"].unique()))

        self.camera_filter.clear()
        self.camera_filter.addItem("All")
        if "Camera" in df.columns:
            self.camera_filter.addItems(sorted(df["Camera"].unique()))


        # Apply Filters
        mask = (df["Date"] >= pd.to_datetime(self.start_date.date().toString("yyyy-MM-dd"))) & \
               (df["Date"] <= pd.to_datetime(self.end_date.date().toString("yyyy-MM-dd")))

        if self.worker_filter.currentText() != "All":
            mask &= df["Name"] == self.worker_filter.currentText()
        if self.camera_filter.currentText() != "All" and "Camera" in df.columns:
            mask &= df["Camera"] == self.camera_filter.currentText()


        df_filtered = df[mask]

        # Clear old plots
        self.figure.clear()
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)

        # Chart 1: Total time per worker
        df_worker_time = df_filtered.groupby("Name")[["Phone Time (s)", "Laptop Time (s)", "Chair Time (s)"]].sum()
        df_worker_time.plot(kind="bar", stacked=True, ax=ax1, title="Total Activity Time per Worker")

        # Chart 2: Efficiency = Active time ÷ total observed
        df_worker_time["Efficiency"] = (df_worker_time.sum(axis=1) / (df_worker_time.sum(axis=1).max() + 1e-5)) * 100
        df_worker_time["Efficiency"].plot(kind="bar", ax=ax2, color="orange", title="Worker Efficiency (%)")

        # Chart 3: Object usage counts
        df_counts = df_filtered.groupby("Name")[["Phone Count", "Chair Count"]].sum()
        df_counts.plot(kind="bar", ax=ax3, title="Object Usage Count")

        # Chart 4: Trend over time
        df_trend = df_filtered.groupby("Date")[["Phone Time (s)", "Laptop Time (s)", "Chair Time (s)"]].sum()
        df_trend.plot(ax=ax4, marker="o", title="Trend Over Time")

        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ReportDashboard()
    win.show()
    sys.exit(app.exec_())
