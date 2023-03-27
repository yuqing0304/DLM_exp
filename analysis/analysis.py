import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def plot_acc(df_combined, seed_num, lang, agent):

    train_df = df_combined[df_combined["mode"] == "train"]
    train_stats = train_df.groupby("epoch")["acc"].agg(['mean', 'std']).reset_index()

    test_df = df_combined[df_combined["mode"] == "test"]
    test_stats = test_df.groupby("epoch")["acc"].agg(['mean', 'std']).reset_index()

    # ========================plot the figure for train & test========================
    plt.figure(figsize=(8, 6))

    plt.plot(train_stats["epoch"], train_stats["mean"], label='train')
    plt.fill_between(train_stats["epoch"], train_stats["mean"] + train_stats["std"], train_stats["mean"] - train_stats["std"], alpha=0.3) # np.arange(len(train_stats["epoch"]))

    plt.plot(test_stats["epoch"], test_stats["mean"], label='test')
    plt.fill_between(test_stats["epoch"], test_stats["mean"] + test_stats["std"], test_stats["mean"] - test_stats["std"], alpha=0.3) # np.arange(len(test_stats["epoch"]))

    if agent == "Speaking": 
        test_multi_stats = test_df.groupby("epoch")["multi_acc"].agg(['mean', 'std']).reset_index()
        plt.plot(test_multi_stats["epoch"], test_multi_stats["mean"], label='permissive_test')
        plt.fill_between(test_multi_stats["epoch"], test_multi_stats["mean"] - test_multi_stats["std"], test_multi_stats["mean"] + test_multi_stats["std"], alpha=0.2) # np.arange(len(test_multi_stats["epoch"]))

    plt.grid(True, linestyle='--', color='gray', alpha=0.5)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    if seed_num > 20:  # a single random seed 
        plt.title(f'Train & Test {agent} Accuracy (Random Seed {seed_num}) for {lang}')    
    elif seed_num <= 20: # several seeds
        plt.title(f'Train & Test {agent} Accuracy (Averaged over {seed_num} Random Seeds) for {lang}')
    plt.legend()

    for spine in plt.gca().spines.values():
        spine.set_edgecolor('none')
        spine.set_alpha(0)

    plt.show()




def data_acc(folder_path, file, plt_one=True):
    """
    Combine the training logs of different random seeds with the same parameters
    Arguments:
        folder_path -- /training_log 
        file -- a list of training log file names
        plt_one -- plot individually for each random seed
    
    Returns:
        df_combined, seed_num, lang, agent
    """
    if plt_one: 
        for f in file:

            if f.split("_")[2] != "split*":
                lang = f.split("_")[1][4:] + "_" + f.split("_")[2]
            else: 
                lang = f.split("_")[1][4:]

            if f.split("_")[-1][:-4] == "spk":
                agent = "Speaking"
            elif f.split("_")[-1][:-4] == "lst":
                agent = "Listening" 
            elif f.split("_")[-1][:-4] == "comm":
                agent = "Communication"

            file_list = glob.glob(folder_path + f)

            for i, file_path in enumerate(file_list):
                seed_num = file_path.split("_")[-2][4:]
                seed_num = int(seed_num)
                df = pd.read_csv(file_list[i], sep="\t")
                if df[df["mode"] == "train"].iloc[-1]["acc"] < 0:
                    continue
                else:               
                    df_combined = df
                    plot_acc(df_combined, seed_num, lang, agent)  

    else: 
        for f in file:

            if f.split("_")[2] != "split*":
                lang = f.split("_")[1][4:] + "_" + f.split("_")[2]
            else: 
                lang = f.split("_")[1][4:]

            if f.split("_")[-1][:-4] == "spk":
                agent = "Speaking"
            elif f.split("_")[-1][:-4] == "lst":
                agent = "Listening" 
            elif f.split("_")[-1][:-4] == "comm":
                agent = "Communication"

            file_list = glob.glob(folder_path + f)
            df_list = []
            seed_num = 0

            for i, file_path in enumerate(file_list):
                df = pd.read_csv(file_path, sep="\t")
                ## check the acc value at the last epoch
                if df[df["mode"] == "train"].iloc[-1]["acc"] < 0:
                    continue
                else:
                    seed_num += 1 
                    df_list.append(df)

            df_combined = pd.concat(df_list, axis=0)
            plot_acc(df_combined, seed_num, lang, agent)




# ====================Word order====================

def result_order(df, df_merged, epoch_num):

    results = {}
    value_counts = df_merged['meaning_type'].value_counts()

    percent_s = value_counts["initial_subject_modified"] / len(df) * 100
    percent_o = value_counts["initial_object_modified"] / len(df) * 100


    # add column 'word_order' with content based on conditions
    df_merged.loc[df_merged['message'] == df_merged['output_VSO'], 'word_order'] = 'SO'
    df_merged.loc[df_merged['message'] == df_merged['output_VOS'], 'word_order'] = 'OS'
    df_merged.loc[(df_merged['message'] != df_merged['output_VSO']) & (df_merged['message'] != df_merged['output_VOS']), 'word_order'] = 'none'



    # overall SO/OS order
    order_value_counts = df_merged['word_order'].value_counts()

    if "SO" in order_value_counts:
        percent_so = order_value_counts["SO"] / len(df_merged) * 100
    else: 
        percent_so = 0

    if "OS" in order_value_counts:
        percent_os = order_value_counts["OS"] / len(df_merged) * 100
    else:
        percent_os = 0

    if "none" in order_value_counts:
        percent_none = order_value_counts["none"] / len(df_merged) * 100
    else:
        percent_none = 0


    # SO/OS order conditioned on whether S/O is modified 
    ism_rows = df_merged[df_merged['meaning_type'] == 'initial_subject_modified']

    percent_so_sm = (ism_rows['word_order'] == 'SO').mean() * 100
    percent_os_sm = (ism_rows['word_order'] == 'OS').mean() * 100
    percent_none_sm = (ism_rows['word_order'] == 'none').mean() * 100


    iom_rows = df_merged[df_merged['meaning_type'] == 'initial_object_modified']

    percent_so_om = (iom_rows['word_order'] == 'SO').mean() * 100
    percent_os_om = (iom_rows['word_order'] == 'OS').mean() * 100
    percent_none_om = (iom_rows['word_order'] == 'none').mean() * 100

    if epoch_num == 0:
        epoch = "epoch1"
    else:
        epoch = f"epoch{epoch_num}0"
    results[epoch] = {
    "percent_s": percent_s,
    "percent_o": percent_o,
    "percent_so": percent_so,
    "percent_os": percent_os,
    "percent_none": percent_none,
    "percent_so_sm": percent_so_sm,
    "percent_os_sm": percent_os_sm,
    "percent_none_sm": percent_none_sm,
    "percent_so_om": percent_so_om,
    "percent_os_om": percent_os_om,
    "percent_none_om": percent_none_om}

    return results




def plot_order(results):
    epochs = [key[5:] for key in results.keys()]
    percent_so_sm = [results[key]['percent_so_sm'] for key in results.keys()]
    percent_os_sm = [results[key]['percent_os_sm'] for key in results.keys()]
    percent_none_sm = [results[key]['percent_none_sm'] for key in results.keys()]

    percent_so_om = [results[key]['percent_so_om'] for key in results.keys()]
    percent_os_om = [results[key]['percent_os_om'] for key in results.keys()]
    percent_none_om = [results[key]['percent_none_om'] for key in results.keys()]

    fig, ax = plt.subplots(figsize=(10, 8))
    bar_width = 0.15
    opacity = 0.8
    index = np.arange(len(epochs))

    rects1 = plt.bar(index, percent_so_sm, bar_width, alpha=opacity, color="#ADD8D6", label='SO_SM')
    rects2 = plt.bar(index + bar_width, percent_os_sm, bar_width, alpha=opacity, color= "#FFB6B1", label='OS_SM')
    rects5 = plt.bar(index + 2*bar_width, percent_none_sm, bar_width, alpha=opacity, color= "#808080", label='other_SM')

    rects3 = plt.bar(index + 3*bar_width, percent_so_om, bar_width, alpha=opacity, color= "#FFB6E1", label='SO_OM')
    rects4 = plt.bar(index + 4*bar_width, percent_os_om, bar_width, alpha=opacity, color="#ADD8A6", label='OS_OM')
    rects6 = plt.bar(index + 5*bar_width, percent_none_om, bar_width, alpha=opacity, color="#808088", label='other_OM')

    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.title('Verb-Initial Language')
    plt.xticks(index + 1.5*bar_width, epochs)
    plt.legend()

    plt.tight_layout()
    # print(f'{i}_initial_simple_order_{dirs[i]}.jpg')
    # plt.savefig(f'{i}_initial_simple_order_{dirs[i]}.jpg', format='jpg', dpi=300)
    plt.show()




def data_order(folder_path, ref, file, plt_one=True):
    if plt_one: 
        for f in file:
            dirs = glob.glob(os.path.join(folder_path, f), recursive=True)

            dump = {}
            for i, dir in enumerate(dirs):
                dump[i] = [[] for _ in range(7)]
                        
            for i, dir in enumerate(dirs):
                agent_num = i  # seed number
                myresults = {}

                for epoch_num in range(7):
                    if epoch_num == 0:
                        pattern = f'*epoch1.txt'
                    else:
                        pattern = f'*epoch{epoch_num}0.txt'
                    txt_files = []
                    # for dir_path in dirs[i]:    
                    #     txt_files.extend(glob.glob(os.path.join(dir_path, pattern)))
                    txt_files.extend(glob.glob(os.path.join(dir, pattern)))
                    dump[i][epoch_num] = pd.concat((pd.read_csv(file_path, delimiter='\t') for file_path in txt_files), ignore_index=True).iloc[:, 1:-1]
                    #  if plt_one==True: txt_files only contain one element, dump[i][epoch_num] one df
                    dump[i][epoch_num]["message"] = dump[i][epoch_num]["message"].str.split().str.slice(stop=7).str.join(' ')
                    # dump files 
                    # if epoch_num == 0:
                    #     file_name = f'agent{agent_num}_dump_epoch1.csv'
                    # else:
                    #     file_name = f'agent{agent_num}_dump_epoch{epoch_num}0.csv'
                    df = dump[i][epoch_num]
                    # # df.to_csv(os.path.join("agent_dump/", file_name), index=False)

                    df_merged = pd.merge(dump[i][epoch_num], ref, on='meaning', how='left')

                    myresults.update(result_order(df, df_merged, epoch_num))
                plot_order(myresults)
    else:
        for f in file:
            dirs = glob.glob(os.path.join(folder_path, f), recursive=True)

            dump = [[] for _ in range(7)]
            myresults = {}
                        
            for i, dir in enumerate(dirs):
                agent_num = i  # seed number

                for epoch_num in range(7):
                    if epoch_num == 0:
                        pattern = f'*epoch1.txt'
                    else:
                        pattern = f'*epoch{epoch_num}0.txt'
                    txt_files = []
                    for dir_path in dirs:    
                        txt_files.extend(glob.glob(os.path.join(dir_path, pattern)))

                    dump[epoch_num] = pd.concat((pd.read_csv(file_path, delimiter='\t') for file_path in txt_files), ignore_index=True).iloc[:, 1:-1]
                    dump[epoch_num]["message"] = dump[epoch_num]["message"].str.split().str.slice(stop=7).str.join(' ')
            
            for epoch_num, _ in enumerate(dump):
                df_merged = pd.merge(dump[epoch_num], ref, on='meaning', how='left')
                # dump files 
                # if epoch_num == 0:
                #     file_name = f'agent{agent_num}_dump_epoch1.csv'
                # else:
                #     file_name = f'agent{agent_num}_dump_epoch{epoch_num}0.csv'
                df = dump[epoch_num]
            #     # # df.to_csv(os.path.join("agent_dump/", file_name), index=False)

                myresults.update(result_order(df, df_merged, epoch_num))
            plot_order(myresults)
