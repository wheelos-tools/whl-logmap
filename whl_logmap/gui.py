import sys
import math
from typing import Optional, List, Tuple, Dict
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QAction
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
import pyqtgraph as pg

class Node:
    """Doubly linked list node, storing point coordinates and pointers to previous and next nodes"""
    def __init__(self, data: Tuple[float, float]):
        self.data: Tuple[float, float] = data
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class PointList:
    """Doubly linked list managing point data, supporting fast index access"""
    def __init__(self):
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.size: int = 0
        self.nodes_by_index: Dict[int, Node] = {}  # Mapping from index to node

    def append(self, data: Tuple[float, float]) -> None:
        """Add a new node at the end of the list"""
        new_node = Node(data)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.nodes_by_index[self.size] = new_node
        self.size += 1

    def insert_between(self, data: Tuple[float, float], prev_node: Node, next_node: Node) -> Node:
        """Insert a new node between prev_node and next_node, update index"""
        new_node = Node(data)
        new_node.prev, new_node.next = prev_node, next_node
        if prev_node:
            prev_node.next = new_node
        if next_node:
            next_node.prev = new_node

        # Update the index of subsequent nodes in nodes_by_index
        index = list(self.nodes_by_index.keys())[list(self.nodes_by_index.values()).index(prev_node)] + 1
        for i in range(self.size, index, -1):
            self.nodes_by_index[i] = self.nodes_by_index[i-1]
        self.nodes_by_index[index] = new_node
        self.size += 1
        return new_node

    def remove_node(self, node: Node) -> None:
        """Remove the specified node, update index"""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        # Update nodes_by_index
        index = list(self.nodes_by_index.keys())[list(self.nodes_by_index.values()).index(node)]
        del self.nodes_by_index[index]
        for i in range(index, self.size-1):
            self.nodes_by_index[i] = self.nodes_by_index[i+1]
        del self.nodes_by_index[self.size-1]
        self.size -= 1

    def get_points_list(self) -> List[Tuple[float, float]]:
        """Return a list of coordinates of all points"""
        points = []
        current = self.head
        while current:
            points.append(current.data)
            current = current.next
        return points

    def get_segments_list(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return a list of endpoint coordinates of all segments"""
        segments = []
        current = self.head
        while current and current.next:
            segments.append((current.data, current.next.data))
            current = current.next
        return segments


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Data Visualization")
        self.setGeometry(100, 100, 800, 600)

        # Initialize core components
        self.central_widget = pg.PlotWidget()
        self.setCentralWidget(self.central_widget)
        self.point_list = PointList()
        self.scatter_plot: Optional[pg.ScatterPlotItem] = None
        self.line_plot: Optional[pg.PlotDataItem] = None
        self.selected_point_node: Optional[Node] = None
        self.current_mode: str = "none"  # State variable: none/add/delete/move
        self.highlighted_index: Optional[int] = None  # Index of the highlighted point

        # Initialize menu and status bar
        self.create_actions()
        self.create_menus()
        self.statusBar().showMessage("Ready")

    # ----------------------------------------------
    # Menu and action initialization
    # ----------------------------------------------
    def create_actions(self):
        """Create menu actions"""
        self.load_action = self._create_action("Load Data", self.load_data)
        self.add_action = self._create_action("Add Point", lambda: self.set_mode("add"))
        self.delete_action = self._create_action("Delete Point", lambda: self.set_mode("delete"))
        self.move_action = self._create_action("Move Point", lambda: self.set_mode("move"))
        self.undo_action = self._create_action("Undo", self.undo, shortcut="Ctrl+Z")
        self.redo_action = self._create_action("Redo", self.redo, shortcut="Ctrl+Y")

    def create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.load_action)
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction(self.add_action)
        edit_menu.addAction(self.delete_action)
        edit_menu.addAction(self.move_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)

    def _create_action(self, text: str, slot, shortcut: Optional[str] = None) -> QAction:
        """Create and return a QAction"""
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    # ----------------------------------------------
    # Mode management and status prompt
    # ----------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Set the current operation mode and update the status bar"""
        self.current_mode = mode
        status_messages = {
            "none": "Ready",
            "add": "Add Point Mode: Click between segments to add a point",
            "delete": "Delete Point Mode: Click a point to delete",
            "move": "Move Point Mode: Drag a point to move"
        }
        self.statusBar().showMessage(status_messages.get(mode, "Ready"))
        self._update_cursor(mode)

    def _update_cursor(self, mode: str) -> None:
        """Update the mouse cursor based on the mode"""
        cursors = {
            "none": Qt.ArrowCursor,
            "add": Qt.CrossCursor,
            "delete": Qt.ForbiddenCursor,
            "move": Qt.OpenHandCursor
        }
        self.central_widget.setCursor(QCursor(cursors.get(mode, Qt.ArrowCursor)))

    # ----------------------------------------------
    # Data loading and saving
    # ----------------------------------------------
    def load_data(self) -> None:
        """Load point data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Point Data File", "", "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                self.point_list = PointList()  # Clear existing data
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        x, y = map(float, line.split(','))
                        self.point_list.append((x, y))
                    except ValueError:
                        QMessageBox.warning(
                            self, "Format Error", f"Unable to parse line {line_num}: {line}"
                        )
                        self.point_list = PointList()
                        break
                self.update_plot()
                self.statusBar().showMessage(f"File loaded: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file: {str(e)}")

    # ----------------------------------------------
    # Plot update and performance optimization
    # ----------------------------------------------
    def update_plot(self) -> None:
        """Update scatter plot and line plot"""
        points = self.point_list.get_points_list()
        segments = self.point_list.get_segments_list()

        # Update scatter plot
        if points:
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            brushes = ['r' if i == self.highlighted_index else 'b' for i in range(len(points))]
            if self.scatter_plot:
                self.scatter_plot.setData(x=x, y=y, brush=brushes)
            else:
                self.scatter_plot = pg.ScatterPlotItem(
                    x=x, y=y, size=10, pen=pg.mkPen(None), brush=brushes
                )
                self.central_widget.addItem(self.scatter_plot)
        elif self.scatter_plot:
            self.central_widget.removeItem(self.scatter_plot)
            self.scatter_plot = None

        # Update line plot
        if segments:
            lines = []
            for seg in segments:
                lines.append([seg[0], seg[1]])
            if self.line_plot:
                self.line_plot.setData(pos=lines, connect='all', pen='r')
            else:
                self.line_plot = pg.PlotDataItem(pos=lines, connect='all', pen='r')
                self.central_widget.addItem(self.line_plot)
        elif self.line_plot:
            self.central_widget.removeItem(self.line_plot)
            self.line_plot = None

    # ----------------------------------------------
    # Mouse event handling (split logic)
    # ----------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            mouse_pos = self._get_mouse_pos(event)
            if self.current_mode == "delete":
                self._handle_delete(mouse_pos)
            elif self.current_mode == "move":
                self._handle_move_start(mouse_pos)
            elif self.current_mode == "add":
                self._handle_add_point(mouse_pos)

    def mouseMoveEvent(self, event):
        if self.current_mode == "move" and self.selected_point_node:
            new_pos = self._get_mouse_pos(event)
            self._update_point_position(new_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_mode == "move":
            self.selected_point_node = None
            self.highlighted_index = None
            self.update_plot()

    def _get_mouse_pos(self, event) -> Tuple[float, float]:
        """Get the position of the mouse in the plot coordinate system"""
        pos = self.central_widget.plotItem.vb.mapSceneToView(event.pos())
        return (pos.x(), pos.y())

    # ----------------------------------------------
    # Delete mode handling
    # ----------------------------------------------
    def _handle_delete(self, mouse_pos: Tuple[float, float]) -> None:
        """Handle click events in delete mode"""
        node = self._find_nearest_node(mouse_pos, threshold=10)
        if node:
            self.point_list.remove_node(node)
            self.update_plot()
            self.statusBar().showMessage("Point deleted")
        else:
            self.statusBar().showMessage("No nearby point found")
        self.set_mode("none")

    # ----------------------------------------------
    # Move mode handling
    # ----------------------------------------------
    def _handle_move_start(self, mouse_pos: Tuple[float, float]) -> None:
        """Handle the start of move mode"""
        self.selected_point_node = self._find_nearest_node(mouse_pos, threshold=10)
        if self.selected_point_node:
            points = self.point_list.get_points_list()
            self.highlighted_index = points.index(self.selected_point_node.data)
            self.update_plot()
            self.statusBar().showMessage("Drag to move point")
        else:
            self.statusBar().showMessage("No nearby point found")
            self.set_mode("none")

    def _update_point_position(self, new_pos: Tuple[float, float]) -> None:
        """Update the position of the selected point"""
        if not self.selected_point_node:
            return
        try:
            x, y = new_pos
            if math.isinf(x) or math.isinf(y) or math.isnan(x) or math.isnan(y):
                return
            self.selected_point_node.data = (x, y)
            self.update_plot()
        except:
            pass

    # ----------------------------------------------
    # Add mode handling
    # ----------------------------------------------
    def _handle_add_point(self, mouse_pos: Tuple[float, float]) -> None:
        """Handle click events in add mode"""
        nearest_segment = self._find_nearest_segment(mouse_pos, threshold=10)
        if nearest_segment is not None:
            prev_node, next_node = nearest_segment
            self.point_list.insert_between(mouse_pos, prev_node, next_node)
            self.update_plot()
            self.statusBar().showMessage("Point added")
        else:
            self.statusBar().showMessage("No nearby segment found")
        self.set_mode("none")

    # ----------------------------------------------
    # Geometric calculation utility methods
    # ----------------------------------------------
    @staticmethod
    def _distance_to_segment(
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float]
    ) -> float:
        """Calculate the shortest distance from a point to a segment"""
        line_vec = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
        point_vec = (point[0] - seg_start[0], point[1] - seg_start[1])
        line_len_sq = line_vec[0]**2 + line_vec[1]**2

        if line_len_sq == 0:
            return math.hypot(point_vec[0], point_vec[1])

        t = max(0, min(1, (point_vec[0]*line_vec[0] + point_vec[1]*line_vec[1]) / line_len_sq))
        projection = (
            seg_start[0] + t * line_vec[0],
            seg_start[1] + t * line_vec[1]
        )
        return math.hypot(point[0] - projection[0], point[1] - projection[1])

    def _find_nearest_node(
        self,
        mouse_pos: Tuple[float, float],
        threshold: float = 10
    ) -> Optional[Node]:
        """Find the node closest to the mouse position"""
        min_dist_sq = threshold ** 2
        nearest_node = None
        for i, point in enumerate(self.point_list.get_points_list()):
            dx = point[0] - mouse_pos[0]
            dy = point[1] - mouse_pos[1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_node = self.point_list.nodes_by_index[i]
        return nearest_node

    def _find_nearest_segment(
        self,
        mouse_pos: Tuple[float, float],
        threshold: float = 10
    ) -> Optional[Tuple[Node, Node]]:
        """Find the segment closest to the mouse position"""
        points = self.point_list.get_points_list()
        min_distance = float('inf')
        nearest_segment = None

        for i in range(len(points) - 1):
            seg_start = points[i]
            seg_end = points[i+1]
            distance = self._distance_to_segment(mouse_pos, seg_start, seg_end)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                prev_node = self.point_list.nodes_by_index[i]
                next_node = self.point_list.nodes_by_index[i+1]
                nearest_segment = (prev_node, next_node)

        return nearest_segment

    # ----------------------------------------------
    # Undo/redo functionality (reserved interface)
    # ----------------------------------------------
    def undo(self):
        """Undo the last operation"""
        # To be implemented: use command pattern to record operation history
        pass

    def redo(self):
        """Redo the last undone operation"""
        # To be implemented
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
