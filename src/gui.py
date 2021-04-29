import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot, QObject, pyqtSignal, QThread
from main import backend

import time
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
        self.query = dict()
        with open(os.path.join(os.path.dirname(sys.executable), 'config.json'), 'r') as f:
            self.query = json.load(f)
        self.backgroundThread = QThread()
        self.initUI()

    def onUpdateText(self, text):
        # self.console.append(text)
        self.cursor = self.console.textCursor()
        self.cursor.movePosition(QTextCursor.End)
        self.cursor.insertText(text)
        self.console.setTextCursor(self.cursor)
        self.console.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__

    def initUI(self):
        self.setWindowTitle(self.title)
        layout = QGridLayout(self)
        self.setLayout(layout)

        wkdir_layout = QHBoxLayout()
        label_wkdir = QLabel('Working Directory:')
        label_wkdir.setMinimumWidth(120)
        wkdir_layout.addWidget(label_wkdir)
        ledit_wkdir = QLineEdit(os.path.abspath(self.query['working_directory']))
        wkdir_layout.addWidget(ledit_wkdir)
        btn_wkdir = QPushButton('Select Folder', self)
        btn_wkdir.setMinimumWidth(100)
        btn_wkdir.clicked.connect(lambda: self.openWkdirDialog(ledit_wkdir))
        wkdir_layout.addWidget(btn_wkdir)

        jitter_layout = QHBoxLayout()
        label_jitter = QLabel('Jitter File:')
        label_jitter.setMinimumWidth(120)
        jitter_layout.addWidget(label_jitter)
        ledit_jitter = QLineEdit(os.path.abspath(self.query['jitter_path']))
        jitter_layout.addWidget(ledit_jitter)
        btn_jitter = QPushButton('Select File', self)
        btn_jitter.setMinimumWidth(100)
        btn_jitter.clicked.connect(lambda: self.openJitterDialog(ledit_jitter))
        jitter_layout.addWidget(btn_jitter)

        nof_layout = QHBoxLayout()
        label_nof = QLabel('Num of Frames:')
        label_nof.setMinimumWidth(120)
        nof_layout.addWidget(label_nof)
        ledit_nof = QLineEdit(str(self.query['number_of_frames']))
        nof_layout.addWidget(ledit_nof)

        layout.addLayout(wkdir_layout, 0, 0)
        layout.addLayout(jitter_layout, 1, 0)
        layout.addLayout(nof_layout, 2, 0)

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
            self.query['number_of_frames'] = int(ledit_nof.text())
            with open(os.path.join(os.path.dirname(sys.executable), 'config.json'), 'w') as f:
                json.dump(self.query, f, indent=2)
            self.backend_request = self.backend_worker(self.query)
            self.backend_request.start()
            btn_submit.setEnabled(False)
            self.backend_request.finished.connect(lambda: btn_submit.setEnabled(True))
            os.chdir(os.path.dirname(sys.executable))

    def openWkdirDialog(self, line):
        options = QFileDialog.Options(QFileDialog.ShowDirsOnly)
        path = QFileDialog.getExistingDirectory(self, "Select Folder", ".")
        print(f'Working directory folder: {path}')
        line.setText(path)

    def openJitterDialog(self, line):
        options = QFileDialog.Options()
        path = QFileDialog.getOpenFileName(self, "Select File", ".")
        print(f'Jitter Path: {path[0]}')
        line.setText(path[0])

    class backend_worker(QThread):
        def __init__(self, query):
            QThread.__init__(self)
            self.query = query
        def __del__(self):
            self.wait()
        def run(self):
            print('----------------------------------------------')
            print('Calling backend')
            backend(self.query)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
