import csv
import datetime
import os
import paramiko
import sys
import time

import bayes_opt
import connect_ssh
import job_command
import pandas as pd


class Share:
    def __init__(self, LOCAL_DIR, REMOTE_DIR, WORK_DIR, JOB_USER, INIT_NUM, BO_NUM):
        self.LOCAL_DIR = self.normalize_dir(LOCAL_DIR)
        self.REMOTE_DIR = self.normalize_dir(REMOTE_DIR)
        self.WORK_DIR = self.normalize_dir(WORK_DIR)
        self.JOB_USER = JOB_USER
        self.INIT_NUM = int(INIT_NUM)
        self.BO_NUM = int(BO_NUM)
        self.all_env = {
            "LOCAL_DIR": self.LOCAL_DIR,
            "REMOTE_DIR": self.REMOTE_DIR,
            "WORK_DIR": self.WORK_DIR,
            "JOB_USER": self.JOB_USER,
            "INIT_NUM": self.INIT_NUM,
            "BO_NUM": self.BO_NUM,
        }

        self.df_with_energy = pd.DataFrame()
        self.df_error = pd.DataFrame()
        self.df_without_energy = pd.DataFrame()
        self.pkey = None

    def normalize_dir(self, dir):
        if dir[0] != "/":
            dir = "/" + dir
        if dir[-1] == "/":
            dir = dir[:-1]

        return dir

    def main(self):
        data_path = f"{self.LOCAL_DIR}{self.WORK_DIR}/calc/data.csv"
        # データの読み込み
        df = pd.read_csv(data_path, engine="python", encoding="utf-8")

        # 'result'列を浮動小数点数型に変換
        df["result"] = df["result"].astype(float)

        # IDをインデックスに設定
        df = df.set_index("ID")

        # 'result'列にデータがある行のうち、値が0未満の行のみを抽出
        self.df_with_energy = df.dropna(subset=["result"]).query("result < 0")

        # 'result'列にデータがある行のうち、値が0以上の行のみを抽出
        self.df_error = df.dropna(subset=["result"]).query("result >= 0")

        # 'result'列にデータがない行のみを抽出
        self.df_without_energy = df[df["result"].isnull()]

        # print("w/ ene", df_with_energy, "\n", "error", df_error, "\n", "w/o ene", df_without_energy)
        KEY_FILENAME = f"{self.LOCAL_DIR}/high-entropy-oxide/share/busseiken_private"
        self.pkey = paramiko.RSAKey(filename=KEY_FILENAME)

        while len(self.df_with_energy) < self.INIT_NUM + self.BO_NUM:
            # 新たな候補を生成
            job_list, calc_dir = self.create_candidates()
            test_list = job_list

            # ジョブを実行
            self.create_jobs(job_list, calc_dir)

            self.update_data(job_list, calc_dir)

            # dfへの書き込み確認
            print(
                f"df_with_energy: len={len(self.df_with_energy)}\n{self.df_with_energy.tail(5)}"
            )
            print(f"df_error: len={len(self.df_error)}\n{self.df_error.tail(5)}")

            # 候補dfの削除確認
            missing_indexes = [
                i for i in test_list if i not in self.df_without_energy.index
            ]
            if not missing_indexes:
                print("Deletion successful")
            else:
                print("Deletion failed")

            # 一分待つ
            time.sleep(10)

    def create_candidates(self):
        # index.pyを実行する
        job_list, calc_dir = bayes_opt.make_candidates(
            self.all_env, self.df_with_energy, self.df_without_energy
        )
        return job_list, calc_dir

    def create_jobs(self, job_list, calc_dir):
        # job_command.pyを実行する
        command_list = job_command.make_command(self.all_env, job_list, calc_dir)
        ssh_results = connect_ssh.execute(self.all_env, command_list, self.pkey)
        if ssh_results[0][1]:
            print(ssh_results[0][1])
            sys.exit(1)

    def check_completed(self, local_id, local_calc_dir):
        result = 0
        # エラーの種類の定義
        ERROR_ERROR_FILE_PRESENT = 1
        ERROR_TIME_OVER = 2
        ERROR_OVER_THRESHOLD = 3
        ERROR_OVER_LIMIT_CALCULATION_TIMES = 4
        ERROR_FE_DAT_EMPTY = 5

        THRESHOLD = 0.03
        LIMIT_LINE_LENGTH = 2000

        FE_DAT_DIRECTORY = f"{self.REMOTE_DIR}{self.WORK_DIR}/calc/results"

        # エラーファイルの有無を確認
        # エラーファイルは〇〇の時に生成される
        if (
            connect_ssh.execute(
                self.all_env,
                [
                    f"test -e {FE_DAT_DIRECTORY}{local_calc_dir}/{local_id}/error.txt; echo $?"
                ],
                self.pkey,
            )[0][0].strip()
            == "0"
        ):  # エラーファイルがある場合
            result = ERROR_ERROR_FILE_PRESENT
        else:  # エラーファイルがない場合
            if (
                connect_ssh.execute(
                    self.all_env,
                    [
                        f"test -e {FE_DAT_DIRECTORY}{local_calc_dir}/{local_id}/fe.dat; echo $?"
                    ],
                    self.pkey,
                )[0][0].strip()
                == "1"
            ):  # fe.datファイルがない場合
                # print("fe.datファイルが見つかりません")
                if self.check_expired(local_calc_dir, local_id):
                    result = ERROR_TIME_OVER

            else:  # fe.datファイルがある場合
                fe_dat_last_line_list = connect_ssh.execute(
                    self.all_env,
                    [f"tail -n 1 {FE_DAT_DIRECTORY}{local_calc_dir}/{local_id}/fe.dat"],
                    self.pkey,
                )[0][0].split()

                if not fe_dat_last_line_list:
                    result = ERROR_FE_DAT_EMPTY
                else:
                    # 最適化回数(1列目)を定義
                    opt_count = int(fe_dat_last_line_list[0])
                    # 閾値上下の判断基準(2列目)を定義
                    final_values = float(fe_dat_last_line_list[1])
                    # エネルギーの最終値(3列目)を定義
                    final_energy = fe_dat_last_line_list[2]

                    # 閾値以下のエネルギーが得られたか確認
                    if final_values >= THRESHOLD:  # 閾値が判断基準値以上の場合
                        if (
                            opt_count == LIMIT_LINE_LENGTH
                        ):  # 最適化回数がLIMIT_LINE_LENGTHに達した場合
                            result = ERROR_OVER_LIMIT_CALCULATION_TIMES
                        else:  # 最適化回数がLIMIT_LINE_LENGTHに達していながら閾値が基準以上の場合
                            # エネルギーがTHRESHOLDより大きい場合
                            result = ERROR_OVER_THRESHOLD
                    else:  # エネルギーがTHRESHOLDより小さい場合
                        result = final_energy
        return result

    # job.txt内の計算終了時刻と現在時刻を比較
    def check_expired(self, local_calc_dir, local_id):
        # job.txtの有無を確認
        # if not os.path.isfile(
        #     f"{self.REMOTE_DIR}{self.WORK_DIR}/calc/results{local_calc_dir}/{local_id}/job.txt"
        # ):# job.txtがない場合
        #     return False
        if (
            connect_ssh.execute(
                self.all_env,
                [
                    f"test -e {self.REMOTE_DIR}{self.WORK_DIR}/calc/results{local_calc_dir}/{local_id}/job.txt; echo $?"
                ],
                self.pkey,
            )[0][0].strip()
            == "1"
        ):  # job.txtファイルがない場合
            return False

        # elseはいらない？
        # job.txtを読取モードで開く
        # with open(
        #     f"{self.REMOTE_DIR}{self.WORK_DIR}/calc/results{local_calc_dir}/{local_id}/job.txt",
        #     "r",
        # ) as f:
        #     #'date'行(2行目)を確認
        #     date = f.readlines()[1].strip()
        date = connect_ssh.execute(
            self.all_env,
            [
                f"sed -n '2p' {self.REMOTE_DIR}{self.WORK_DIR}/calc/results{local_calc_dir}/{local_id}/job.txt"
            ],
            self.pkey,
        )[0][0].strip()
        # print(date)
        date_formated = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        # 現在時刻を取得
        now = datetime.datetime.now()
        # job.txt内の時刻と現在時刻を比較
        if date_formated < now:
            return True
        else:
            return False

    # 「job_list」操作対象のPOSCARのIDが格納されたリスト
    def update_data(self, local_job_list, local_calc_dir):
        DATA_CSV = f"{self.LOCAL_DIR}{self.WORK_DIR}/calc/data.csv"

        while len(local_job_list) > 0:
            for id in local_job_list:
                result_energy = self.check_completed(id, local_calc_dir)
                if result_energy == float(
                    0
                ):  # check_completedの返り値(エネルギー)が0、即ち計算中の場合
                    pass
                else:
                    # 抽出
                    df_to_add = self.df_without_energy[
                        self.df_without_energy.index == id
                    ]

                    # エネルギーの列を追加
                    df_to_add.loc[id, "result"] = float(result_energy)

                    # 削除
                    # エネルギーを計算できたIDはジョブリスト内から削除
                    local_job_list.pop(local_job_list.index(id))
                    # 未計算テーブル内から対象IDのPOSCARを削除(抽出分以外で再定義)
                    self.df_without_energy = self.df_without_energy[
                        self.df_without_energy.index != id
                    ]

                    # 追加
                    # スパコンから得たエネルギーをjob.txtへ書き込む
                    # textファイルの有無を確認
                    # if os.path.isfile(
                    #     f"{ERROR_FILE_DIRECTORY}{local_calc_dir}/{id}/job.txt"
                    # ):
                    #     with open(
                    #         f"{ERROR_FILE_DIRECTORY}{local_calc_dir}/{id}/job.txt", "r+"
                    #     ) as jobfile:
                    #         rows_text = jobfile.readlines()
                    #         if len(rows_text) == 2:
                    #             rows_text.append(f"{result_energy}\n")
                    #             jobfile.seek(0)
                    #             jobfile.writelines(rows_text)
                    #         else:
                    #             print("%sのjob.txtの行数が正しくありません" % id)
                    #             sys.exit(1)
                    # else:
                    #     print("%sのjob.txtファイルが見つかりません" % id)
                    #     sys.exit(1)

                    # csvファイルの有無を確認
                    if os.path.isfile(f"{DATA_CSV}"):
                        pass
                    else:
                        print("data.csvファイルが見つかりません")
                        sys.exit(1)

                    # 全POSCARファイルが記載されているCSVファイルに追記
                    # 追記&読取モードでファイルを開く
                    with open(
                        f"{DATA_CSV}",
                        "r+",
                    ) as csvfile:
                        reader_csv = csv.reader(csvfile)
                        rows_csv = list(reader_csv)
                        writer_csv = csv.writer(csvfile)
                        header_csv = rows_csv[0]
                        for row in rows_csv[1:]:
                            if int(row[0]) == id:  # IDが一致した場合
                                # 行の情報としてエネルギーを追加
                                row[-1] = str(result_energy)
                                break
                        # TODO: より軽い処理を考える

                        # 更新内容を踏まえて書き直す
                        csvfile.seek(0)
                        writer_csv.writerow(header_csv)
                        writer_csv.writerows(rows_csv[1:])

                    # result_energyが正ならエラーテーブルに追加、負なら計算済テーブルに追加
                    if float(result_energy) > 0:
                        self.df_error = pd.concat([self.df_error, df_to_add])
                    else:
                        self.df_with_energy = pd.concat(
                            [self.df_with_energy, df_to_add]
                        )

            time.sleep(10)
            continue


def make_instance(LOCAL_DIR, REMOTE_DIR, WORK_DIR, JOB_USER, INIT_NUM, BO_NUM):
    share_instance = Share(LOCAL_DIR, REMOTE_DIR, WORK_DIR, JOB_USER, INIT_NUM, BO_NUM)
    share_instance.main()
