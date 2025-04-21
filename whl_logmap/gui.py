#!/usr/bin/env python3
import sys
import math
from typing import Optional, List, Tuple, Dict

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QAction
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
import pyqtgraph as pg

# 通过 import 调用车道生成模块中的接口
from map_gen_single_lane import (generate_lane_from_points, generate_and_save_map,
                                   load_map_from_file, extract_lane_geometry)

# ---------------------------
# 双向链表存储点数据（原有代码）
# ---------------------------
class Node:
    """双向链表节点，存储点坐标及前后节点指针"""
    def __init__(self, data: Tuple[float, float]):
        self.data: Tuple[float, float] = data
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class PointList:
    """管理点数据的双向链表，支持快速索引访问"""
    def __init__(self):
        self.head: Optional[Node] = None
        self.tail: Optional[Node] = None
        self.size: int = 0
        self.nodes_by_index: Dict[int, Node] = {}  # 索引到节点的映射

    def append(self, data: Tuple[float, float]) -> None:
        """在链表尾部添加新节点"""
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
        """在 prev_node 与 next_node 之间插入新节点，更新索引"""
        new_node = Node(data)
        new_node.prev, new_node.next = prev_node, next_node
        if prev_node:
            prev_node.next = new_node
        if next_node:
            next_node.prev = new_node

        # 更新后续节点的索引映射
        index = list(self.nodes_by_index.keys())[list(self.nodes_by_index.values()).index(prev_node)] + 1
        for i in range(self.size, index, -1):
            self.nodes_by_index[i] = self.nodes_by_index[i-1]
        self.nodes_by_index[index] = new_node
        self.size += 1
        return new_node

    def remove_node(self, node: Node) -> None:
        """删除指定节点，更新索引"""
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev

        index = list(self.nodes_by_index.keys())[list(self.nodes_by_index.values()).index(node)]
        del self.nodes_by_index[index]
        for i in range(index, self.size-1):
            self.nodes_by_index[i] = self.nodes_by_index[i+1]
        del self.nodes_by_index[self.size-1]
        self.size -= 1

    def get_points_list(self) -> List[Tuple[float, float]]:
        """返回所有点的坐标列表"""
        points = []
        current = self.head
        while current:
            points.append(current.data)
            current = current.next
        return points

    def get_segments_list(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """返回所有线段端点坐标的列表"""
        segments = []
        current = self.head
        while current and current.next:
            segments.append((current.data, current.next.data))
            current = current.next
        return segments

# ---------------------------
# 主窗口
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Data & Lane Visualization")
        self.setGeometry(100, 100, 800, 600)

        # 初始化核心组件（设置深色背景、显示网格、锁定长宽比，确保俯视效果）
        self.central_widget = pg.PlotWidget()
        self.central_widget.setBackground('k')
        self.central_widget.showGrid(x=True, y=True, alpha=0.3)
        self.central_widget.setAspectLocked(True)
        self.setCentralWidget(self.central_widget)
        
        self.point_list = PointList()
        self.scatter_plot: Optional[pg.ScatterPlotItem] = None
        self.line_plot: Optional[pg.PlotDataItem] = None
        self.selected_point_node: Optional[Node] = None
        self.current_mode: str = "none"  # 操作模式：none/add/delete/move
        self.highlighted_index: Optional[int] = None  # 高亮点索引

        # 用于显示生成或加载的车道线
        self.lane_center_plot: Optional[pg.PlotDataItem] = None
        self.lane_left_plot: Optional[pg.PlotDataItem] = None
        self.lane_right_plot: Optional[pg.PlotDataItem] = None

        # 初始化菜单与状态栏
        self.create_actions()
        self.create_menus()
        self.statusBar().showMessage("Ready")

    # ----------------------------------------------
    # 菜单和动作初始化
    # ----------------------------------------------
    def create_actions(self):
        """创建菜单动作"""
        self.load_data_action = self._create_action("Load Data", self.load_data)
        self.load_map_action = self._create_action("Load Map", self.load_map)
        self.add_action = self._create_action("Add Point", lambda: self.set_mode("add"))
        self.delete_action = self._create_action("Delete Point", lambda: self.set_mode("delete"))
        self.move_action = self._create_action("Move Point", lambda: self.set_mode("move"))
        self.undo_action = self._create_action("Undo", self.undo, shortcut="Ctrl+Z")
        self.redo_action = self._create_action("Redo", self.redo, shortcut="Ctrl+Y")
        self.generate_lane_action = self._create_action("Generate Lane", self.generate_lane)
        self.save_map_action = self._create_action("Save Map", self.save_map)

    def create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.load_data_action)
        file_menu.addAction(self.load_map_action)
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction(self.add_action)
        edit_menu.addAction(self.delete_action)
        edit_menu.addAction(self.move_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.generate_lane_action)
        edit_menu.addAction(self.save_map_action)

    def _create_action(self, text: str, slot, shortcut: Optional[str] = None) -> QAction:
        """创建并返回一个 QAction"""
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        return action

    # ----------------------------------------------
    # 模式管理及状态提示
    # ----------------------------------------------
    def set_mode(self, mode: str) -> None:
        """设置当前操作模式并更新状态栏提示"""
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
        """根据模式更新鼠标样式"""
        cursors = {
            "none": Qt.ArrowCursor,
            "add": Qt.CrossCursor,
            "delete": Qt.ForbiddenCursor,
            "move": Qt.OpenHandCursor
        }
        self.central_widget.setCursor(QCursor(cursors.get(mode, Qt.ArrowCursor)))

    # ----------------------------------------------
    # 数据加载（加载中心线点数据）
    # ----------------------------------------------
    def load_data(self) -> None:
        """加载点数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Point Data File", "", "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                self.point_list = PointList()  # 清空已有数据
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
    # 地图加载（加载生成的地图文件并显示车道几何）
    # ----------------------------------------------
    def load_map(self) -> None:
        """加载地图文件（.txt 或 .bin），解析后提取车道几何并显示"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Map File", "", "Map Files (*.txt *.bin);;All Files (*)"
        )
        if not file_path:
            return
        try:
            m = load_map_from_file(file_path)
            center_line, left_boundary, right_boundary = extract_lane_geometry(m)
            # 移除已有车道图形
            if self.lane_center_plot:
                self.central_widget.removeItem(self.lane_center_plot)
            if self.lane_left_plot:
                self.central_widget.removeItem(self.lane_left_plot)
            if self.lane_right_plot:
                self.central_widget.removeItem(self.lane_right_plot)
            # 绘制车道中心线（黄色虚线）和左右边界（白色实线）
            self.lane_center_plot = pg.PlotDataItem(
                x=[pt[0] for pt in center_line],
                y=[pt[1] for pt in center_line],
                pen=pg.mkPen(color='y', style=Qt.DashLine, width=2)
            )
            self.lane_left_plot = pg.PlotDataItem(
                x=[pt[0] for pt in left_boundary],
                y=[pt[1] for pt in left_boundary],
                pen=pg.mkPen(color='w', width=2)
            )
            self.lane_right_plot = pg.PlotDataItem(
                x=[pt[0] for pt in right_boundary],
                y=[pt[1] for pt in right_boundary],
                pen=pg.mkPen(color='w', width=2)
            )
            self.central_widget.addItem(self.lane_center_plot)
            self.central_widget.addItem(self.lane_left_plot)
            self.central_widget.addItem(self.lane_right_plot)
            self.central_widget.autoRange()
            self.statusBar().showMessage("Map loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load map: {str(e)}")

    # ----------------------------------------------
    # 图形更新（绘制点数据及连线）
    # ----------------------------------------------
    def update_plot(self) -> None:
        """更新散点图和线段图显示"""
        points = self.point_list.get_points_list()
        segments = self.point_list.get_segments_list()

        # 更新散点图
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

        # 更新连线图（连接各点）
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
    # 鼠标事件处理
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
        """将鼠标位置转换为图形坐标系下的坐标"""
        pos = self.central_widget.plotItem.vb.mapSceneToView(event.pos())
        return (pos.x(), pos.y())

    # ----------------------------------------------
    # 删除模式处理
    # ----------------------------------------------
    def _handle_delete(self, mouse_pos: Tuple[float, float]) -> None:
        """删除模式下的点击事件处理"""
        node = self._find_nearest_node(mouse_pos, threshold=10)
        if node:
            self.point_list.remove_node(node)
            self.update_plot()
            self.statusBar().showMessage("Point deleted")
        else:
            self.statusBar().showMessage("No nearby point found")
        self.set_mode("none")

    # ----------------------------------------------
    # 移动模式处理
    # ----------------------------------------------
    def _handle_move_start(self, mouse_pos: Tuple[float, float]) -> None:
        """移动模式起始，选择最近的点"""
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
        """更新选中点的位置"""
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
    # 添加模式处理
    # ----------------------------------------------
    def _handle_add_point(self, mouse_pos: Tuple[float, float]) -> None:
        """添加模式下的点击事件处理"""
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
    # 几何计算辅助方法
    # ----------------------------------------------
    @staticmethod
    def _distance_to_segment(
        point: Tuple[float, float],
        seg_start: Tuple[float, float],
        seg_end: Tuple[float, float]
    ) -> float:
        """计算点到线段的最短距离"""
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
        """查找离鼠标位置最近的节点"""
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
        """查找离鼠标位置最近的线段"""
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
    # Undo/Redo（预留接口）
    # ----------------------------------------------
    def undo(self):
        """撤销上一步操作"""
        # TODO: 使用命令模式记录操作历史
        pass

    def redo(self):
        """重做上一步撤销操作"""
        # TODO: 实现重做功能
        pass

    # ----------------------------------------------
    # 集成的车道生成功能（调用外部模块）
    # ----------------------------------------------
    def generate_lane(self):
        """根据当前加载的点数据生成单车道地图，并在上位机视图中显示道路中心线、左边界和右边界（俯视图效果）"""
        points = self.point_list.get_points_list()
        if len(points) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 2 points to generate lane.")
            return

        try:
            center_line, left_boundary, right_boundary = generate_lane_from_points(
                points, lane_width=3.3, step=1.0, sample_interval=0.5
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate lane: {str(e)}")
            return

        if self.lane_center_plot:
            self.central_widget.removeItem(self.lane_center_plot)
        if self.lane_left_plot:
            self.central_widget.removeItem(self.lane_left_plot)
        if self.lane_right_plot:
            self.central_widget.removeItem(self.lane_right_plot)

        self.lane_center_plot = pg.PlotDataItem(
            x=[pt[0] for pt in center_line],
            y=[pt[1] for pt in center_line],
            pen=pg.mkPen(color='y', style=Qt.DashLine, width=2)
        )
        self.lane_left_plot = pg.PlotDataItem(
            x=[pt[0] for pt in left_boundary],
            y=[pt[1] for pt in left_boundary],
            pen=pg.mkPen(color='w', width=2)
        )
        self.lane_right_plot = pg.PlotDataItem(
            x=[pt[0] for pt in right_boundary],
            y=[pt[1] for pt in right_boundary],
            pen=pg.mkPen(color='w', width=2)
        )
        self.central_widget.addItem(self.lane_center_plot)
        self.central_widget.addItem(self.lane_left_plot)
        self.central_widget.addItem(self.lane_right_plot)
        self.central_widget.autoRange()
        self.statusBar().showMessage("Lane generated")

    def save_map(self):
        """调用外部模块接口，根据当前点数据生成地图，并保存为 .txt 和 .bin 文件"""
        points = self.point_list.get_points_list()
        if len(points) < 2:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 2 points to save map.")
            return

        txt_file, _ = QFileDialog.getSaveFileName(
            self, "Save Map as Text", "", "Text Files (*.txt);;All Files (*)"
        )
        if not txt_file:
            return

        bin_file, _ = QFileDialog.getSaveFileName(
            self, "Save Map as Binary", "", "Binary Files (*.bin);;All Files (*)"
        )
        if not bin_file:
            return

        try:
            generate_and_save_map(points, txt_file, bin_file, extra_extension=0.0)
            self.statusBar().showMessage("Map saved successfully")
            QMessageBox.information(self, "Save Map", f"Map saved:\nText: {txt_file}\nBinary: {bin_file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save map: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
