import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal
from main import backend

import os
import json


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PF32 Analysis Tool'
        self.setWindowIcon(QIcon())
        self.setGeometry(50, 50, 500, 300)
        sys.stdout = Stream(newText=self.onUpdateText)
        self.query = {'working_directory': None}
        os.chdir(os.path.dirname(__file__))
        with open('config.json', 'r') as f:
            self.query = json.load(f)
        self.initUI()

    def onUpdateText(self, text):
        # self.console.append(text)
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__

    def initUI(self):
        self.setWindowTitle(self.title)
        layout = QGridLayout(self)
        self.setLayout(layout)

        wkdir_layout = QHBoxLayout()
        label_wkdir = QLabel('Working Directory:')
        label_wkdir.setMinimumWidth(100)
        wkdir_layout.addWidget(label_wkdir)
        ledit_wkdir = QLineEdit(self.query['working_directory'])
        wkdir_layout.addWidget(ledit_wkdir)
        btn_wkdir = QPushButton('Select Folder', self)
        btn_wkdir.clicked.connect(lambda: self.openWkdirDialog(ledit_wkdir))
        wkdir_layout.addWidget(btn_wkdir)

        jitter_layout = QHBoxLayout()
        label_jitter = QLabel('Jitter File:')
        label_jitter.setMinimumWidth(100)
        jitter_layout.addWidget(label_jitter)
        ledit_jitter = QLineEdit(self.query['jitter_path'])
        jitter_layout.addWidget(ledit_jitter)
        btn_jitter = QPushButton('Select File', self)
        btn_jitter.clicked.connect(lambda: self.openJitterDialog(ledit_jitter))
        jitter_layout.addWidget(btn_jitter)

        layout.addLayout(wkdir_layout, 0, 0)
        layout.addLayout(jitter_layout, 1, 0)

        btn_submit = QPushButton('Submit', self)
        btn_submit.clicked.connect(lambda: wrap_query())
        layout.addWidget(btn_submit)

        self.console = QTextEdit(self)
        self.console.setReadOnly(True)
        layout.addWidget(self.console)

        self.show()

        def wrap_query():
            self.query['working_directory'] = ledit_wkdir.text()
            self.query['jitter_path'] = ledit_jitter.text()
            with open('config.json', 'w') as f:
                json.dump(self.query, f, indent=2)
            self.backend_request()
            os.chdir(os.path.dirname(__file__))

    def openWkdirDialog(self, line):
        options = QFileDialog.Options(QFileDialog.ShowDirsOnly)
        path = QFileDialog.getExistingDirectory(self, "Select Folder", ".")
        print(f'Working directory folder: {path}')
        line.setText(path)

    def openJitterDialog(self, line):
        options = QFileDialog.Options()
        path = QFileDialog.getOpenFileName(self, "Select File", ".csv")
        print(f'Jitter Path: {path}')
        line.setText(path)

    def backend_request(self):
        print('----------------------------------------------')
        print('Calling backend')
        backend(self.query)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
