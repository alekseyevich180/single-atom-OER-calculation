def make_command(all_env, job_list, calc_dir):
    REMOTE_DIR = all_env["REMOTE_DIR"]
    WORK_DIR = all_env["WORK_DIR"]
    JOB_USER = all_env["JOB_USER"]
    # 作業ディレクトリ
    job_dir = f"{REMOTE_DIR}{WORK_DIR}/calc/results{calc_dir}"
    conditions_dir = f"{REMOTE_DIR}{WORK_DIR}/calc/conditions"
    share_dir = f"{REMOTE_DIR}/high-entropy-oxide/share"
    POSCARs_DIR = f"{REMOTE_DIR}{WORK_DIR}/structures/one_million_models/POSCARs"
    sh_file = "HEO_default.sh" if JOB_USER != "test" else "HEO_test.sh"
    command_list = []

    for i in job_list:
        # コマンドを作成
        command = f"""
        check_dir() {{
            if [ ! -d "{job_dir}" ]; then
                echo "Error: {job_dir} does not exist" >&2
                exit 1
            fi
        }}
        check_dir &&
        cd {job_dir} &&
        mkdir {i} &&
        cp {conditions_dir}/* {i}/ &&
        cp {POSCARs_DIR}/POSCAR_{i} {i}/POSCAR &&
        cp {share_dir}/{sh_file} {i}/{JOB_USER}_{i}.sh &&
        cd {i} &&
        qsub {JOB_USER}_{i}.sh
        """
        command_list.append(command)
    return command_list
