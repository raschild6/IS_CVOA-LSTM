from CVOA.Individual import Individual
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.offsetbox import AnchoredText
import time as t
import tkinter as tk

import numpy as np
import sys as sys
import random as random
from DEEP_LEARNING.LSTM import fit_lstm_model, getMetrics_denormalized, resetTF


class CVOA:

    min_x = 0
    max_x = 10
    infected_index = 1
    individual_text = AnchoredText('', loc=10)
    values_text = AnchoredText('', loc=6)
    stat_text = AnchoredText('', loc=1)
    actual_text = AnchoredText('', prop=dict(fontsize=14, bbox=None), loc=8)
    process_text = AnchoredText('Starting...', prop=dict(fontsize=16), loc=4)
    # upper-center loc=9
    def on_launch(self):
        #Set up plot
        self.figure, (self.ax1,self.ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
        self.linesY, = self.ax2.plot([],[], 'yo')
        self.linesG, = self.ax2.plot([],[], 'go')
        self.linesR, = self.ax2.plot([],[], 'ro')
        #Autoscale on unknown axis and known lims on the other
        self.ax2.set_autoscaley_on(True)
        self.ax2.set_xlim(self.min_x, self.max_x)
        self.ax1.axis('off')
        self.ax2.set_xlabel('Epoches ( ∀ epoches: Death - Infected - Recovered )' , fontsize=12, labelpad=15)
        self.ax2.set_ylabel('MAPE', fontsize=12, labelpad=15)
        self.ax2.legend(labels=['Infected', 'Recovered', 'Deaths'])
        self.ax2.xaxis.set_ticks(np.arange(0, 10, 1.0))

        self.ax1.add_artist(self.values_text)
        self.ax1.add_artist(self.individual_text)
        self.ax1.add_artist(self.stat_text)
        self.ax1.add_artist(self.actual_text)
        self.ax1.add_artist(self.process_text)

        # place a text box in upper left
        self.upgrade_values()

        # Max size of windows plot
        mng = plt.get_current_fig_manager()
        mng.window.state("zoomed")

    def plot_points(self, xdata_r, ydata_r, xdata_y, ydata_y, xdata_g, ydata_g):
        #Update data (with the new _and_ the old points)
        self.linesR.set_xdata(xdata_r)
        self.linesR.set_ydata(ydata_r)
        self.linesY.set_xdata(xdata_y)
        self.linesY.set_ydata(ydata_y)
        self.linesG.set_xdata(xdata_g)
        self.linesG.set_ydata(ydata_g)
        #Need both of these in order to rescale
        self.ax2.relim()
        self.ax2.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def upgrade_values(self):                           # TODO: PROBLEMA sistemare allineamento!
        self.values_text.remove()
        textstr = ("Infected:         {}\n" +
                   "\nRecovered:     {}\n" + 
                   "\nDeaths:           {}").format(str(len(self.infected)), str(len(self.recovered)), str(len(self.deaths))) #.expandtabs()

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, pad=1.5)
        #self.ax1.text(0.05, 0.95, textstr, transform=self.ax1.transAxes, fontsize=16, verticalalignment='top', bbox=props)
        self.values_text = AnchoredText(textstr, prop=dict(fontsize=16, bbox=props), loc=6)
        self.ax1.add_artist(self.values_text)


    def upgrade_Individual(self, individual):                           # PROBLEMA sistemare allineamento!
        self.individual_text.remove()
        textstr = str(individual)
        self.individual_text = AnchoredText(textstr,  prop=dict(fontsize=25), loc=10)
        self.ax1.add_artist(self.individual_text)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def upgrade_Stat(self, mape):                           # PROBLEMA sistemare allineamento!
        self.stat_text.remove()
        textstr = ("BEST_MAPE:           {}\n" +
                 "Recovered/Infected:  {:.2f} %").format(mape, 100 * len(self.recovered) / len(self.infected))

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=1.0, pad=0.8)
        self.stat_text = AnchoredText(textstr,  prop=dict(fontsize=16, bbox=props), loc=1)
        self.ax1.add_artist(self.stat_text)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def upgrade_Actual(self, mape):                           # PROBLEMA sistemare allineamento!
        self.actual_text.remove()
        textstr = ("Actual_MAPE: {}").format(mape)

        props = dict(boxstyle='round', facecolor='white', alpha=1.0, pad=0.5)
        self.actual_text = AnchoredText(textstr,  prop=dict(fontsize=13, bbox=props), loc=8)
        self.ax1.add_artist(self.actual_text)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def upgrade_Process(self, process):                           # PROBLEMA sistemare allineamento!
        self.process_text.remove()
        textstr = ("{}").format(process)

        props = dict(boxstyle='round', facecolor='white', alpha=1.0, pad=0.55)
        self.process_text = AnchoredText(textstr,  prop=dict(fontsize=16, bbox=props), loc=4)
        self.ax1.add_artist(self.process_text)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def display_time():
        current_time = tm.strftime('%H:%M:%S')
        clock_label['text'] = current_time
        self.label = tk.Label(self, text="", width=10)

    bestSolution = None
    bestModel = None
    MIN_SPREAD = 0
    MAX_SPREAD = 5
    MIN_SUPERSPREAD = 6
    MAX_SUPERSPREAD = 25
    P_TRAVEL = 0.1
    P_REINFECTION = 0.01
    SUPERSPREADER_PERC = 0.04
    DEATH_PERC = 0.5 
    point_x = 0.0
    xdata_r = []
    ydata_r = []
    xdata_y = []
    ydata_y = []
    xdata_g = []
    ydata_g = []

    def __init__(self, size_fixed_part, min_size_var_part, max_size_var_part, fixed_part_max_values, var_part_max_value, max_time, xtrain, ytrain, xval, yval, pred_horizon=1, epochs=10, batch=512, scaler=None):
        self.infected = []
        self.recovered = []
        self.deaths = []
        self.min_size_var_part = min_size_var_part
        self.max_size_var_part = max_size_var_part
        self.size_fixed_part = size_fixed_part
        self.fixed_part_max_values = fixed_part_max_values
        self.var_part_max_value = var_part_max_value
        self.max_time = max_time
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xval = xval
        self.yval = yval
        self.pred_horizon = pred_horizon
        self.batch = batch
        self.epochs = epochs
        self.scaler = scaler

        self.infected_index = 1

    def calcSearchSpaceSize (self):
        """
        :return: Total search space
        """
        t = 1
        res = 0
        # Part 1. Fixed part possible combinations.
        for i in range(len(self.fixed_part_max_values)):
            t *= self.fixed_part_max_values[i]
        res += t * self.max_size_var_part
        # Part 2. Var part possible combinations.
        res *= pow(self.var_part_max_value, self.max_size_var_part)
        return res

    def propagateDisease(self):
        new_infected_list = []
        self.infected_index = 1
        # Step 1. Assess fitness for each individual.
        for x in self.infected:
            x.fitness, model = self.fitness(x)
            # If x.fitness is NaN, move from infected list to deaths lists
            if np.isnan(x.fitness):
                self.deaths.append(x)
                self.infected.remove(x)
        
        
        self.point_x += 0.2
        for d in self.deaths:
            if(np.isnan(d.fitness)):
                continue
            else:
                self.xdata_r.append(self.point_x)
                #ydata_r.append(point_y)
                self.ydata_r.append(d.fitness)
                #self.point_y += 1
        self.point_x += 0.2
        #self.point_y = 1s
        for i in self.infected:
            self.xdata_y.append(self.point_x)
            self.ydata_y.append(i.fitness)
            #point_y += 1
        self.point_x += 0.2
        #point_y = 1
        for r in self.recovered:
            self.xdata_g.append(self.point_x)
            self.ydata_g.append(r.fitness)
            #point_y += 1
            
        self.plot_points(self.xdata_r, self.ydata_r, self.xdata_y, self.ydata_y, self.xdata_g, self.ydata_g)
        #point_y = 1
        self.point_x += 0.4

        # Step 2. Sort the infected list by fitness (ascendent).
        self.infected = sorted(self.infected, key=lambda i: i.fitness)
        # Step 3. Update best global solution, if proceed.
        if self.bestSolution.fitness is None or self.infected[0].fitness < self.bestSolution.fitness:
            self.bestModel = model
            model.save("bestModel.h5")
            self.bestSolution = deepcopy(self.infected[0])

        resetTF()  # Release GPU memory
        # Step 4. Assess indexes to point super-spreaders and deaths parts of the infected list.
        if len(self.infected) == 1:
            idx_super_spreader = 1
        else:
            idx_super_spreader = self.SUPERSPREADER_PERC * len(self.infected)
        if len(self.infected) == 1:
            idx_deaths = sys.maxsize
        else:
            idx_deaths = len(self.infected) - (self.DEATH_PERC * len(self.infected))

        # Step 5. Disease propagation.
        i = 0
        for x in self.infected:
            # Step 5.1 If the individual belongs to the death part, then die!
            if i >= idx_deaths:
                self.deaths.append(x)
                self.infected.remove(x)
            else:
                # Step 5.2 Determine the number of new infected individuals.
                if i < idx_super_spreader:  # This is the super-spreader!
                    ninfected = self.MIN_SUPERSPREAD + random.randint(0, self.MAX_SUPERSPREAD - self.MIN_SUPERSPREAD)
                else:
                    ninfected = random.randint(0, self.MAX_SPREAD)
                # Step 5.3 Determine whether the individual has traveled
                if random.random() < self.P_TRAVEL:
                    traveler = True
                else:
                    traveler = False
                # Step 5.4 Determine the travel distance, which is how far is the new infected individual.
                if traveler:
                    travel_distance = -1  # The value -1 is to indicate that all the individual elements can be affected.  # TODO: -1 indicate nmutated (numero mutazioni) = randint(0, max -1)
                else:
                    travel_distance = 1  # TODO: 1 indicate nmutated = 1,
                # Step 5.5 Infect!!
                for j in range(ninfected):
                    new_infected = x.infect(travel_distance=travel_distance)  # new_infected = infect(x, travel_distance)
                    if new_infected not in self.deaths and new_infected not in self.infected and new_infected not in new_infected_list and new_infected not in self.recovered:
                        new_infected_list.append(new_infected)
                    elif new_infected in self.recovered and new_infected not in new_infected_list:
                        if random.random() < self.P_REINFECTION:
                            new_infected_list.append(new_infected)
                            self.recovered.remove(new_infected)
            i+=1
        # Step 6. Add the current infected individuals to the recovered list.
        self.recovered.extend(self.infected)
        # Step 7. Update the infected list with the new infected individuals.
        self.infected = new_infected_list

    def run(self):
        
        plt.ion()

        epidemic = True
        time = 0
        # Step 1. Infect to Patient Zero
        # pz = Individual.random(size_fixed_part=self.size_fixed_part, min_size_var_part=self.min_size_var_part,
        #                   max_size_var_part=self.max_size_var_part, fixed_part_max_values=self.fixed_part_max_values,
        #                   var_part_max_value=self.var_part_max_value)
        # custom pz
        pz = Individual(self.size_fixed_part, self.min_size_var_part, self.max_size_var_part,
                        self.fixed_part_max_values, self.var_part_max_value)
        pz.fixed_part = [4, 0, 4]
        pz.var_part = [7, 6, 0, 8]
        self.infected.append(pz)

        # start graphic plot
        self.on_launch()
        self.xdata_r = []
        self.ydata_r = []
        self.xdata_y = []
        self.ydata_y = []
        self.xdata_g = []
        self.ydata_g = []
        #point_x = 0.4
        self.point_x = 0.0
        #point_y = 1
        #xdata_y.append(point_x)
        #ydata_y.append(point_y)
        #self.plot_points(xdata_r, ydata_r, xdata_y, ydata_y, xdata_g, ydata_g)
        #point_x += 0.6
        self.upgrade_Individual(pz)

        print("Patient Zero: " + str(pz) + "\n")
        self.bestSolution = deepcopy(pz)
        # Step 2. The main loop for the disease propagation
        total_ss = self.calcSearchSpaceSize()
        while epidemic and time < self.max_time:

            self.upgrade_Process('Epoch: {} - Creating new generation...'.format(time))

            self.propagateDisease()
            dmse, dmape = getMetrics_denormalized(model=self.bestModel, xval=self.xval, yval=self.yval, batch=self.batch, scaler=self.scaler)
            print("Iteration ", (time + 1))
            #print("Best fitness so far: ", "{:.4f}".format(self.bestSolution.fitness))
            print("Best one --- MAPE: {:.4f} ; MSE: {:.4f} ; INDIVIDUAL: : {} ---"
                  .format(self.bestSolution.fitness, dmse, self.bestSolution))
            print("Infected: {} ; Recovered: {} ; Deaths: {} "
                  .format(str(len(self.infected)), str(len(self.recovered)), str(len(self.deaths))))
            print("Recovered/Infected: {:.4f} %".format(100 * len(self.recovered) / len(self.infected)))
            current_ss = len(self.infected) + len(self.recovered) + len(self.deaths)
            print("Percentage of evaluated individual: {} / {} = {:.4f} %"
                  .format(str(current_ss), str(total_ss), 100 * current_ss / total_ss))
            self.upgrade_Stat("{:.4f}".format(self.bestSolution.fitness))
            if not self.infected:
                epidemic = False
            time += 1
            
            self.upgrade_values()

        return self.bestSolution

    def fitness(self, individual):
        self.upgrade_Process('Evaluating fitness - Infected : {}°'.format(self.infected_index))
        mse, mape, model = fit_lstm_model(xtrain=self.xtrain, ytrain=self.ytrain, xval=self.xval, yval=self.yval,
                                          individual_fixed_part=individual.fixed_part,
                                          individual_variable_part=individual.var_part, scaler=self.scaler,
                                          prediction_horizon=self.pred_horizon, epochs=self.epochs, batch=self.batch)
        print("{} --- MSE: {:.4f} ; MAPE: {:.4f} ; INDIVIDUAL: {} ---".format(self.infected_index, mse, mape, str(individual)))
        self.infected_index += 1
        self.upgrade_Individual(individual)
        self.upgrade_Actual("{:.4f}".format(mape))
        return mape.numpy(), model