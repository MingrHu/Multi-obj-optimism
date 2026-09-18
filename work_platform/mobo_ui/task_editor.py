"""Editable DEFORM task definitions used by the desktop workbench."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


def _item(value: Any = "", *, checked: bool | None = None) -> QTableWidgetItem:
    item = QTableWidgetItem(str(value))
    if checked is not None:
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setText("")
    return item


class DefinitionTable(QTableWidget):
    """Small editable table with predictable row selection and column sizing."""

    def __init__(self, headers: list[str]):
        super().__init__(0, len(headers))
        self.setHorizontalHeaderLabels(headers)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setMinimumSectionSize(76)
        self.horizontalHeader().setDefaultSectionSize(132)
        self.horizontalHeader().setStretchLastSection(True)

    def append_row(self, values: list[Any], checks: dict[int, bool] | None = None) -> None:
        row = self.rowCount()
        self.insertRow(row)
        checks = checks or {}
        for column in range(self.columnCount()):
            value = values[column] if column < len(values) else ""
            self.setItem(
                row,
                column,
                _item(value, checked=checks.get(column)) if column in checks else _item(value),
            )

    def remove_selected_rows(self) -> None:
        rows = sorted({index.row() for index in self.selectedIndexes()}, reverse=True)
        for row in rows:
            self.removeRow(row)

    def text(self, row: int, column: int) -> str:
        widget = self.cellWidget(row, column)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        item = self.item(row, column)
        return item.text().strip() if item else ""

    def checked(self, row: int, column: int) -> bool:
        item = self.item(row, column)
        return bool(item and item.checkState() == Qt.CheckState.Checked)

    def set_options(
        self,
        row: int,
        column: int,
        values: list[str],
        current: str,
        *,
        editable: bool = True,
    ) -> None:
        box = QComboBox()
        box.setEditable(editable)
        box.addItems(values)
        if box.findText(current) < 0 and current:
            box.addItem(current)
        box.setCurrentText(current)
        self.setCellWidget(row, column, box)


class TaskDefinitionEditor(QWidget):
    """Edit a task definition without mutating the built-in preset objects."""

    def __init__(self, *, multi: bool):
        super().__init__()
        self.multi = multi
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        identity = QFormLayout()
        self.task_id = QLineEdit()
        self.task_id.setPlaceholderText("英文、数字、点、下划线或短横线；用于任务状态目录")
        self.task_name = QLineEdit()
        self.task_name.setPlaceholderText("界面显示名称")
        identity.addRow("任务 ID", self.task_id)
        identity.addRow("任务名称", self.task_name)
        layout.addLayout(identity)

        self.tabs = QTabWidget()
        if multi:
            self.operation_table = DefinitionTable(
                ["工步名称", "KEY 模板", "继承材料", "启用晶粒", "偏移 X", "偏移 Y", "偏移 Z"]
            )
            self.operation_table.setMinimumHeight(155)
            self.tabs.addTab(
                self._table_panel(
                    self.operation_table,
                    self.add_operation,
                    self.operation_table.remove_selected_rows,
                    browse=self.browse_operation_template,
                ),
                "工步模板",
            )
        else:
            self.operation_table = None

        parameter_headers = ["工步", "参数类型", "作用对象（多个用逗号）", "下限", "上限"]
        self.parameter_table = DefinitionTable(parameter_headers)
        self.parameter_table.setMinimumHeight(155)
        self.tabs.addTab(
            self._table_panel(
                self.parameter_table,
                self.add_parameter,
                self.parameter_table.remove_selected_rows,
            ),
            "输入参数",
        )

        self.target_table = DefinitionTable(
            ["输出列名", "提取目标", "对象", "工步（逗号）", "分量", "全过程", "工件类型"]
        )
        self.target_table.setMinimumHeight(155)
        self.tabs.addTab(
            self._table_panel(
                self.target_table,
                self.add_target,
                self.target_table.remove_selected_rows,
            ),
            "输出目标",
        )
        layout.addWidget(self.tabs)

        hint = QLabel(
            "参数类型使用后端已注册的 KEY 替换名称；作用对象可填写一个对象，或用逗号分隔多个同步对象。"
            "内置预设另存前需要填写新的任务 ID 和名称；用户模板可直接更新。"
            "表格中的当前值会用于本次样本生成、KEY 生成和计算。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("Subtitle")
        layout.addWidget(hint)

    @staticmethod
    def _table_panel(table, add, remove, *, browse=None) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(table)
        actions = QHBoxLayout()
        add_button = QPushButton("+ 添加")
        add_button.clicked.connect(add)
        remove_button = QPushButton("删除选中行")
        remove_button.setObjectName("Danger")
        remove_button.clicked.connect(remove)
        actions.addWidget(add_button)
        actions.addWidget(remove_button)
        if browse is not None:
            browse_button = QPushButton("为选中工步选择 KEY")
            browse_button.clicked.connect(browse)
            actions.addWidget(browse_button)
        actions.addStretch()
        layout.addLayout(actions)
        return panel

    def add_operation(self) -> None:
        if self.operation_table is None:
            return
        index = self.operation_table.rowCount() + 1
        self.operation_table.append_row(
            [f"工步 {index}", "", "", "", "", "", ""],
            checks={2: index > 1, 3: False},
        )

    def browse_operation_template(self) -> None:
        if self.operation_table is None:
            return
        row = self.operation_table.currentRow()
        if row < 0:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self, "选择工步 KEY 模板", self.operation_table.text(row, 1),
            "DEFORM KEY (*.KEY *.key);;所有文件 (*)",
        )
        if selected:
            self.operation_table.setItem(row, 1, _item(selected))

    def add_parameter(self) -> None:
        row = self.parameter_table.rowCount()
        names = self._parameter_names()
        self.parameter_table.append_row(["1", "", "workpiece", "0", "1"])
        self.parameter_table.set_options(row, 1, names, names[0] if names else "")
        self.parameter_table.set_options(row, 2, self._object_names(), "workpiece")

    def add_target(self) -> None:
        row = self.target_table.rowCount()
        targets = self._target_names()
        self.target_table.append_row(
            ["", "", "workpiece", "1", "", "", "generic"], checks={5: False}
        )
        self.target_table.set_options(row, 1, targets, targets[0] if targets else "")
        self.target_table.set_options(row, 2, self._object_names(), "workpiece")
        self.target_table.set_options(row, 6, self._workpiece_types(), "generic")

    def clear_for_new(self) -> None:
        """Reset all fields to a minimal blank definition for user creation."""
        self.task_id.clear()
        self.task_name.clear()
        self.parameter_table.setRowCount(0)
        self.target_table.setRowCount(0)
        if self.operation_table is not None:
            self.operation_table.setRowCount(0)
            self.add_operation()
            self.add_operation()
        self.add_parameter()
        self.add_target()

    def load_definition(self, definition: Any) -> None:
        self.task_id.setText(str(definition.task_id))
        self.task_name.setText(str(definition.name))
        self.parameter_table.setRowCount(0)
        self.target_table.setRowCount(0)
        if self.multi:
            assert self.operation_table is not None
            self.operation_table.setRowCount(0)
            for operation in definition.operations:
                offset = operation.get("position_offset") or ("", "", "")
                self.operation_table.append_row(
                    [
                        operation.get("name", ""),
                        operation.get("template_key", ""),
                        "",
                        "",
                        *offset,
                    ],
                    checks={
                        2: bool(operation.get("inherit_materials", False)),
                        3: bool(operation.get("enable_grain", False)),
                    },
                )
            for operation_index, operation in enumerate(definition.operations, 1):
                for parameter in operation.get("parameters", []):
                    objects = parameter.get("objects") or [parameter.get("object", "")]
                    row = self.parameter_table.rowCount()
                    self.parameter_table.append_row(
                        [
                            operation_index,
                            parameter.get("name", ""),
                            ", ".join(str(value) for value in objects),
                            parameter.get("range", ["", ""])[0],
                            parameter.get("range", ["", ""])[1],
                        ]
                    )
                    self.parameter_table.set_options(
                        row, 1, self._parameter_names(), str(parameter.get("name", ""))
                    )
                    self.parameter_table.set_options(
                        row, 2, self._object_names(), ", ".join(str(value) for value in objects)
                    )
        else:
            for parameter in definition.parameters:
                objects = parameter.get("objects") or [parameter.get("object", "")]
                row = self.parameter_table.rowCount()
                self.parameter_table.append_row(
                    [
                        1,
                        parameter.get("name", ""),
                        ", ".join(str(value) for value in objects),
                        parameter.get("range", ["", ""])[0],
                        parameter.get("range", ["", ""])[1],
                    ]
                )
                self.parameter_table.set_options(
                    row, 1, self._parameter_names(), str(parameter.get("name", ""))
                )
                self.parameter_table.set_options(
                    row, 2, self._object_names(), ", ".join(str(value) for value in objects)
                )
        for target in definition.targets:
            row = self.target_table.rowCount()
            self.target_table.append_row(
                [
                    target.output_name,
                    target.target_name,
                    target.object_name,
                    ", ".join(str(value) for value in target.operation_indices),
                    "" if target.select_component is None else target.select_component,
                    "",
                    target.workpiece_type,
                ],
                checks={5: target.in_progress},
            )
            self.target_table.set_options(
                row, 1, self._target_names(), str(target.target_name)
            )
            self.target_table.set_options(
                row, 2, self._object_names(), str(target.object_name)
            )
            self.target_table.set_options(
                row, 6, self._workpiece_types(), str(target.workpiece_type)
            )

    def build_definition(self, preset: Any, *, workspace: str, template_key: str = "") -> Any:
        from mobo.automation.task_collection import (
            MultiOperationTaskDefinition,
            SingleOperationTaskDefinition,
        )

        task_id, name, root = self._identity(workspace)
        operation_count = self.operation_table.rowCount() if self.operation_table else 1
        parameters_by_operation = self._read_parameters(operation_count)
        targets = self._read_targets(preset, operation_count)
        if self.multi:
            operations = self._read_operations(parameters_by_operation)
            return MultiOperationTaskDefinition(
                task_id=task_id,
                name=name,
                workspace=root,
                operations=tuple(operations),
                targets=tuple(targets),
            )
        if not template_key:
            raise ValueError("请选择单工步 KEY 模板")
        return SingleOperationTaskDefinition(
            task_id=task_id,
            name=name,
            workspace=root,
            template_key=template_key,
            parameters=tuple(parameters_by_operation[1]),
            targets=tuple(targets),
        )

    def _identity(self, workspace: str) -> tuple[str, str, Path]:
        task_id = self.task_id.text().strip()
        name = self.task_name.text().strip()
        if not task_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
            raise ValueError("任务 ID 只能包含英文、数字、点、下划线和短横线")
        if not name:
            raise ValueError("任务名称不能为空")
        if not workspace:
            raise ValueError("请选择任务工作区")
        return task_id, name, Path(workspace)

    def _read_parameters(self, operation_count: int) -> dict[int, list[dict[str, Any]]]:
        from mobo.replacement import registry as replacement_registry

        result: dict[int, list[dict[str, Any]]] = {
            index: [] for index in range(1, operation_count + 1)
        }
        for row in range(self.parameter_table.rowCount()):
            operation_index = self._positive_int(
                self.parameter_table.text(row, 0), f"输入参数第 {row + 1} 行的工步"
            )
            if operation_index > operation_count:
                raise ValueError(f"输入参数第 {row + 1} 行引用了不存在的工步 {operation_index}")
            name = self.parameter_table.text(row, 1)
            if not name or replacement_registry.resolve(name) is None:
                raise ValueError(f"输入参数第 {row + 1} 行使用了未注册的参数类型: {name}")
            objects = self._csv(self.parameter_table.text(row, 2))
            if not objects:
                raise ValueError(f"输入参数第 {row + 1} 行缺少作用对象")
            low = self._number(self.parameter_table.text(row, 3), f"输入参数第 {row + 1} 行下限")
            high = self._number(self.parameter_table.text(row, 4), f"输入参数第 {row + 1} 行上限")
            if low > high:
                raise ValueError(f"输入参数第 {row + 1} 行下限不能大于上限")
            parameter: dict[str, Any] = {"name": name, "range": [low, high]}
            parameter["object" if len(objects) == 1 else "objects"] = (
                objects[0] if len(objects) == 1 else objects
            )
            result[operation_index].append(parameter)
        if not any(result.values()):
            raise ValueError("至少需要配置一个输入参数")
        return result

    def _read_targets(self, preset: Any, operation_count: int) -> list[Any]:
        from mobo.automation.task_collection import TargetDefinition
        from mobo.extraction import registry as extraction_registry

        preset_targets = {
            target.output_name: target for target in getattr(preset, "targets", ())
        }
        targets = []
        for row in range(self.target_table.rowCount()):
            output_name = self.target_table.text(row, 0)
            target_name = self.target_table.text(row, 1)
            object_name = self.target_table.text(row, 2)
            indices = tuple(
                self._positive_int(value, f"输出目标第 {row + 1} 行工步")
                for value in self._csv(self.target_table.text(row, 3))
            )
            self._validate_target_row(
                row, output_name, target_name, object_name, indices, operation_count,
                extraction_registry,
            )
            component_text = self.target_table.text(row, 4)
            preset_target = preset_targets.get(output_name)
            targets.append(TargetDefinition(
                output_name=output_name,
                target_name=target_name,
                object_name=object_name,
                operation_indices=indices,
                select_component=int(component_text) if component_text else None,
                in_progress=self.target_table.checked(row, 5),
                workpiece_type=self.target_table.text(row, 6) or "generic",
                description=preset_target.description if preset_target else "",
                verified=preset_target.verified if preset_target else True,
            ))
        if not targets:
            raise ValueError("至少需要配置一个输出目标")
        return targets

    def _validate_target_row(
        self, row: int, output_name: str, target_name: str, object_name: str,
        indices: tuple[int, ...], operation_count: int, extraction_registry: Any,
    ) -> None:
        if not output_name or not target_name or not object_name or not indices:
            raise ValueError(f"输出目标第 {row + 1} 行的名称、提取目标、对象和工步不能为空")
        if any(index > operation_count for index in indices):
            raise ValueError(f"输出目标第 {row + 1} 行引用了不存在的工步")
        workpiece_type = self.target_table.text(row, 6) or "generic"
        try:
            extraction_registry.resolve(workpiece_type, target_name)
        except KeyError as exc:
            raise ValueError(
                f"输出目标第 {row + 1} 行没有可用的提取能力: "
                f"{workpiece_type}/{target_name}"
            ) from exc

    def _read_operations(
        self, parameters_by_operation: dict[int, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        assert self.operation_table is not None
        if self.operation_table.rowCount() < 2:
            raise ValueError("多工步任务至少需要两个工步")
        operations = []
        for row in range(self.operation_table.rowCount()):
            operation = self._read_operation(row, parameters_by_operation[row + 1])
            operations.append(operation)
        return operations

    def _read_operation(
        self, row: int, parameters: list[dict[str, Any]]
    ) -> dict[str, Any]:
        assert self.operation_table is not None
        operation: dict[str, Any] = {
            "name": self.operation_table.text(row, 0) or f"工步 {row + 1}",
            "template_key": self.operation_table.text(row, 1),
            "parameters": parameters,
        }
        if not operation["template_key"]:
            raise ValueError(f"工步 {row + 1} 缺少 KEY 模板")
        if self.operation_table.checked(row, 2):
            operation["inherit_materials"] = True
        if self.operation_table.checked(row, 3):
            operation["enable_grain"] = True
        offsets = [self.operation_table.text(row, column) for column in (4, 5, 6)]
        if any(offsets):
            if not all(offsets):
                raise ValueError(f"工步 {row + 1} 的位置偏移必须同时填写 X、Y、Z")
            operation["position_offset"] = [
                self._number(value, f"工步 {row + 1} 位置偏移") for value in offsets
            ]
        return operation

    @staticmethod
    def _csv(value: str) -> list[str]:
        return [part.strip() for part in re.split(r"[,，]", value) if part.strip()]

    @staticmethod
    def _number(value: str, label: str) -> float:
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label}必须是数值") from exc

    @staticmethod
    def _positive_int(value: str, label: str) -> int:
        try:
            result = int(value)
        except ValueError as exc:
            raise ValueError(f"{label}必须是正整数") from exc
        if result < 1:
            raise ValueError(f"{label}必须是正整数")
        return result

    @staticmethod
    def _parameter_names() -> list[str]:
        from mobo.replacement import registry

        return registry.keys()

    @staticmethod
    def _object_names() -> list[str]:
        from mobo.automation.config import DeformConfig

        return [*DeformConfig.OBJ_DEF.keys(), "ring_dies"]

    @staticmethod
    def _target_names() -> list[str]:
        from mobo.extraction import registry

        return list(dict.fromkeys(target for _workpiece, target in registry.keys()))

    @staticmethod
    def _workpiece_types() -> list[str]:
        from mobo.extraction import registry

        return list(dict.fromkeys(workpiece for workpiece, _target in registry.keys()))


__all__ = ["DefinitionTable", "TaskDefinitionEditor"]
