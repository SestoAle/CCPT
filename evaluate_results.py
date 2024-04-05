import pdb
import pickle
import argparse
import matplotlib.pyplot as plt
import glob
import pandas as pd
import re
import json

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiments, we will plot more experiments in the same plot (for each value)
    parser.add_argument("-rs", "--runs", default="td3bc")  # Name of the experiments
    parser.add_argument("-f", "--folder", default="arrays")  # Name of the Folder
    parser.add_argument("-ma", "--moving-average", default=1000, type=int)  # COM value for rewards moving average
    parser.add_argument("-e", "--episodes", default=None, type=int) # Number of episodes to consider 

    parser.add_argument('-t', '--with-timesteps', action='store_true')
    parser.set_defaults(with_timesteps=False)

    args = parser.parse_args()

    # Get the real name of the files
    runs = args.runs
    runs = runs.replace(' ', '')
    runs = runs.replace('.json', '')
    runs = runs.split(",")

    filenames = []
    for run_name in runs:
        path = glob.glob(f"{args.folder}/{run_name}.json")
        for filename in path:
            if 'curriculum.json' in filename:
                continue
            with open(filename, 'r') as f:
                filenames.append(filename)
    print(filenames)

    if len(filenames) == 0:
        raise Exception(f"There are no files in the folder *{args.folder}* with names: {run_name}")

    print(f"Here are the file names to plot: {filenames}")

    # Create a dict for numeric table
    numeric_values = dict()

    # For each file name, load the stats
    stats = []
    for run in filenames:
        # Load stats file
        with open(run, 'rb') as handle:
            try:
                stat = json.load(handle)
                stat['name_run'] = run
                # If there is info, change from dict to list 
                if "info" in stat.keys() and stat["info"] is not None:

                    if "time" in stat["info"][0]:
                        stat["time"] = np.asarray([info["time"] for info in  stat["info"]])
                    
                    if "bugs_found" in stat["info"][0]:
                         stat["bugs_found"] = np.asarray([float(info["bugs_found"][:-1]) for info in  stat["info"]])

                # stat["success_rate"] = np.asarray(stat["episode_rewards"]) > 9
                stats.append(stat)
            except Exception as e:
                print(e)
                continue
    # Group stats per name
    stats_dict = dict()
    filenames = []
    for s in stats:
        name_run = s['name_run']
        name_run = re.sub("x-\d", "x", name_run)
        name_run = re.sub("-\d-", "-", name_run)
        name_run = re.sub("-\d$", "", name_run)
        if name_run in stats_dict:
            stats_dict[name_run].append(s)
        else:
            stats_dict[name_run] = [s]
            name_run = name_run.replace("-no-expert", "")
            name_run = re.sub(r"-exp+$", "", name_run)
            filenames.append(name_run)
    

    # Names of the data to plot and plot titels
    data_names = ['episode_rewards', 'mean_entropies', "bugs_found", "episode_timesteps"]
    title_names = ['Environment Rewards', 'Entropy', "Bugs", "Episode Timesteps"]

    # Plot all data in different figures
    for data_name, title_name in zip(data_names, title_names):
        plt.figure()

        for run_name, stats in stats_dict.items():
            all_datas = []
            all_raw_datas = []

            for s in stats:
                try:
                    data = s[data_name]
                    data = pd.DataFrame({'data': data})
                    all_raw_datas.append(data)
                    data = data.ewm(com=args.moving_average).mean()
                    all_datas.append(data)
                except Exception as e:
                    pass
            
            all_datas = np.asarray(all_datas)
            if len(all_datas.shape) == 1:
                continue
            
            episodes = all_datas.shape[1]
            if args.episodes is not None:
                episodes = args.episodes
            
            # Get the timesteps
            timesteps = stats[0]["episode_timesteps"][:episodes]
            timesteps = np.cumsum(timesteps)

            all_raw_datas[0] = all_raw_datas[0][:episodes]
            all_datas = all_datas[:, :episodes, :]

            # Compute means and stds of runs
            means = np.mean(all_datas, axis=0)
            stds = np.std(all_datas, axis=0)

            plt.title(title_name)

            # Decide whether to print based on episodes or timesteps
            x_axis = np.arange(len(means))
            if args.with_timesteps:
                x_axis = timesteps

            plt.plot(x_axis, means)
            means = np.reshape(means, (-1,))
            stds = np.reshape(stds, (-1,))

            plt.fill_between(x_axis, means - stds, means + stds, alpha=0.5, linewidth=0, label='_nolegend_')
            idx_of_max_mean = np.argmax(means)

            if title_name == "Episode Timesteps":
                pass
            else:
                print(f"Mean of the latest {args.moving_average} episodes of {data_name} for {run_name}: {np.mean(all_raw_datas[0].values[-args.moving_average:])}")

        plt.legend(filenames)

    # This scripts aggregate multiple runs with the same name. We don't have them here in FC, so we take only the 0th index
    for stat in stats_dict.values():
        print(f"Time and Success Rate for run with name: {stat[0]['name_run']}")
        try:
            # Print time passed
            hours, rem = divmod(stat[0]['time'][args.episodes - 1 if args.episodes is not None else -1], 3600)
            minutes, seconds = divmod(rem, 60)
            print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        except Exception:
            print("No time stored for this experiment")

    # Show the images
    plt.show()