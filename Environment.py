from Communication import Communicator
import numpy as np
import random as rd
import time
import collections

RIGHT_WING = 1
LEFT_WING = 3
PWM_WING = 190
FRAMES = 4
default_params = [255, PWM_WING, 0, PWM_WING, 0]
                ##header,right,servo,left,controlmode
receive_byt = 7
#0:header(255固定)
#1:right wing(0~255)
#2:servo: elevator
#3:left wing(0~255)
#4:control mode(1:manual, 0:auto)

class Environment():
    """
    強化学習をする際の環境を設定するクラス
    状態取得、報酬決定、行動内容の決定をしている
    """
    def __init__(self):
        self.communicator = Communicator()
        self.reset()
        self.params_to_send = default_params
        self.state = collections.deque()

    def reset(self):
        """
        状態を格納するデックを作る
        最初に足りない分のフレームの状態を保存する
        """
        self.communicator.start_laz(default_params)
        self.state = collections.deque()
        self.params_to_send = default_params
        for i in range(FRAMES):
            self.params_to_send = default_params
            data, _ = self.communicator.recieve_from_laz(byt=receive_byt)
            self.update(flag=False, data=data)

    def update(self, flag, data):
        """
        状態を更新する
        Args:
            flag ([bool]): True:一番古い状態を消す
            data ([list]): 新しく保存する状態
        """
        if flag:
            self.state.pop()
        self.state.appendleft(data)

    def observe_update_state(self):
        """
        状態を更新した上で状態を確認する
        """
        data, ti = self.communicator.recieve_from_laz(byt=receive_byt)
        self.update(flag=True, data=data)
        return self.state, ti

    def observe_state(self):
        """
        状態を確認する
        """
        return self.state

    def observe_reward(self, data):
        """
        報酬を定義する
        Args:
            data ([list]): 状態（1フレーム）
        Returns:
            [int]: 報酬
        """
        ##報酬の設定
        err = abs(float(data[0][1])-0.0)
        if err < 5:##-5~5
            return 100
        elif err < 10:##-10~10
            return 10
        elif err < 20:##-20~20
            return 5
        else:
            return 0

    def observe_terminal(self):
        """
        ターミナルスイッチの情報を読み取る
        Returns:
            bool: ターミナルスイッチ
        """
        return self.communicator.termination_switch(default_params)

    def excute_action(self, action):
        """
        行動内容を定義する
        行動内容に沿って機体に通信を送る
        Args:
            action ([int]): 行動の番号
        """
        if action == 1:
            self.params_to_send[RIGHT_WING]=PWM_WING+50
            self.params_to_send[LEFT_WING]=PWM_WING-10
        elif action == 2:
            self.params_to_send[RIGHT_WING]=PWM_WING+10
            self.params_to_send[LEFT_WING]=PWM_WING-10
        elif action == 3:
            self.params_to_send[RIGHT_WING]=PWM_WING+20
            self.params_to_send[LEFT_WING]=PWM_WING-10
        elif action == 4:
            self.params_to_send[RIGHT_WING]=PWM_WING
            self.params_to_send[LEFT_WING]=PWM_WING-10
        elif action == 5:
            self.params_to_send[RIGHT_WING]=PWM_WING-180
            self.params_to_send[LEFT_WING]=PWM_WING+60
        else:
            self.params_to_send[RIGHT_WING]=PWM_WING
            self.params_to_send[LEFT_WING]=PWM_WING

        self.communicator.send_to_laz(self.params_to_send)
        time.sleep(0.1)

    def get_sentparam(self):
        """
        機体に送った情報を確認する
        Returns:
            [list]: 送った情報
        """
        return self.params_to_send