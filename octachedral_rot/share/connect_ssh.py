import paramiko


class Share:
    def __init__(self, all_env, command_list, pkey):
        self.all_env = all_env
        self.LOCAL_DIR = all_env["LOCAL_DIR"]
        self.command_list = command_list
        self.pkey = pkey

    def main(self):
        with paramiko.SSHClient() as client:
            HOSTNAME = "genkai.hpc.kyushu-u.ac.jp"
            USERNAME = "ku40000345"
            # PASSWORD = 'password'
            # SSH接続
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # 1. パスワード認証認証方式
            # client.connect(hostname=HOSTNAME, port=22, username=USERNAME, password=PASSWORD)

            # 2. 公開鍵認証方式
            client.connect(
                hostname=HOSTNAME, port=22, username=USERNAME, pkey=self.pkey
            )

            # コマンド実行
            ssh_results = []
            for command in self.command_list:
                # コマンドをまとめて実行
                stdin, stdout, stderr = client.exec_command(command)
                ssh_results.append(
                    [stdout.read().decode("utf-8"), stderr.read().decode("utf-8")]
                )
            stdin.close()
            client.close()
            return ssh_results


def execute(all_env, command_list, pkey):
    share_instance = Share(all_env, command_list, pkey)
    return share_instance.main()
