import tensorflow as tf
import numpy as np
from fcm_libr import *
from copy import deepcopy as dc



class pso:
    #modification of the PSO code that found in:
    #https://pub.towardsai.net/implementing-particle-swarm-optimization-in-tensorflow-b501ca4a3c17
    def __init__(self, fitness_fn, dim,
        pop_size=100,
        n_iter=300,
        b=0.7,
        c1=0.4,
        c2=0.6,
        x_min=-1,
        x_max=1,
        x_offset = 0.7,
        ):
        '''
        fitness_fn: Function to be minimized
        dim: (tuple or list), the shape (FCM are initialized by NxN matrix where N number of concepts)
        pop_size: int, the size of population
        b: float, coefficient for velocity update
        c1,c2: floats, cognitive and social coefficients respectively 
        x_min,m_max: Lower and upper boundaries of weight values
                    (weight matrices values are clipped in those boundaries)
        x_offset: float, a value to be added to/substracted from x_min,x_max respectively 
                during the random initialization
        '''
        self.fitness_fn = fitness_fn
        self.pop_size = pop_size
        self.dim = dim
        self.n_iter = n_iter
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.x_min = x_min  
        self.x_max = x_max
        self.offset = x_offset
        #initialization of FCMs
        self.x = self.build_swarm() 
        self.fit_history = []
        self.min_history = []
        self.v = self.start_velocities()
        self.fix_diag()
        self.input_concepts_indexes = []
        self.output_concepts_indexes = []


    def build_swarm(self):
        """Creates the swarm following the selected initialization method. 
        Returns:
            tf.Tensor: The PSO swarm population.  
        """
        
        self.swarm_size = list(self.dim)
        self.swarm_size.insert(0,self.pop_size)
        
        return np.random.uniform(self.swarm_size, self.x_min + self.offset, self.x_max - self.offset)


    def start_velocities(self):
        """Start the velocities of each particle in the population (swarm). 
        Returns:
            tf.Tensor: The starting velocities.  
        """
        return np.random.uniform(
                self.swarm_size,
                self.x_min + self.offset,
                self.x_max - self.offset
            )
        

    def fix_diag(self):
        '''Convert the diagonal of the weight matrix to 0.'''
        #tensor shape = [population size, rows, columns]
        for i in range(self.x.shape[0]):
            np.fill_diagonal(self.x[i], 0)
            np.fill_diagonal(self.v[i], 0)
    
    def set_initialization(self, weight_matrix):
        '''
        Function for intialization of a weight matrix (such as with a corr matrix)
        Args:
            weight_matrix :  a numpy array matrix for initialization
        '''
        assert weight_matrix.shape == self.dim
        self.x = np.tile(weight_matrix, (self.pop_size, 1, 1))
        self.fix_diag()
    


    def specify_input_concept(self, column_index):
        '''
        To specify concepts that act only as inputs (Columns of that concepts have zero values)
        e.g. Rain cannot be affected by conditions such as plain arrivals, water demand )
            FCM (C1 = Rain, C2 = plain arrivals, C3 = Water demand )
            Set the wight values of other concepts towards rain to 0
                specify_input_concept(column_index = 0)

            column_index: int 
        '''
        if column_index not in self.input_concepts_indexes:
            self.input_concepts_indexes.append(column_index)
        for i in range(self.pop_size):
            self.x[i, :, column_index] = 0
            self.v[i, :, column_index] = 0
        print('Concept of column index {}, was set as input concept.'.format(column_index))



    def specify_output_concept(self, row_index):
        '''
        To specify concepts that act only as outputs. (rows of that concepts have zero values)
        e.g. Water demand doesn't affect concepts such as plain arrivals, rain )
            FCM (C1 = Rain, C2 = plain arrivals, C3 = Water demand )
            Set the wight values of other concepts towards rain to 0
                specify_output_concept(row_index = 2)
            row_index: int
        '''
        if row_index not in self.output_concepts_indexes:
            self.output_concepts_indexes.append(row_index)
        for i in range(self.pop_size):
            self.x[i, row_index, :] = 0
            self.v[i, row_index, :] = 0
        print('Concept of row index {}, was set as output concept.'.format(row_index))


    def get_randoms(self):
        """Generate random values to update the particles' positions. 
        Returns:
            _type_: _description_
        """
        
        random_size = dc(list(self.dim))
        random_size.insert(0, 2)

        return np.random.uniform(0, 1, random_size)

    def update_p_best(self, f_x):
        """Updates the *p-best* positions. 
        """
        self.fit_history.append(np.mean(f_x))
        self.min_history.append(np.min(f_x))
        print(f'minimum error {np.min(f_x)}')
        self.p = np.where(f_x < self.f_p, self.x, self.p) #self.p = personal best positions
        
    def update_g_best(self):
        """Update the *g-best* position. 
        """
        
        self.g = self.p[np.argmin(input=self.f_p)]
        

    def step(self, f_x): 
        """It runs ONE step on the particle swarm optimization. 
        """
        r1, r2 = self.get_randoms()
        self.v = (
            self.b * self.v
            + self.c1 * r1 * (self.p - self.x)
            + self.c2 * r2 * (self.g - self.x)
        )
        self.x = np.clip(self.x + self.v, self.x_min, self.x_max)
        
        self.update_p_best(f_x)
        self.update_g_best()
        print(self.g)


    def fcm_learning(self,data, l =1, fcm_iterations = 20, verbose = -1, exploration_decay = True):
        '''
        FCM learning with PSO.
        Each row is passed as initial state to FCMs that were generated with PSO. 
        The simulation of the FCM is then performed (sigmoid transfer function, stylios inference rule). 
        The error (fitness_fn) of the output of the FCMs and next the next row is then being calculated.
        Particles position is updated based on this error.
        After the given number of iterations or after termination of optimization, the next row is given as initial state...

        data: Numpy array
        l: parameter of sigmoid function
        fcm_iterations: The number of iterations of fcm simulation
        verbose: -1, 0, 1 #to be fixed
        exploration_decay: Boolean. If True b parameter is decayed to allow more exploitation through time. Linear decay

        '''
        #to do: verbose and printing

        self.exploration_decay = exploration_decay
        dataset_length = len(data)
        self.p = self.x
        for i in range(self.n_iter):
            print('Iter {}'.format(i))
            f_x = tf.zeros((self.pop_size,1)) #error at each iter
            for row in range(dataset_length - 1):
                row_tensor = tf.convert_to_tensor(data[row], dtype=tf.float32)
                row_tensor_test = tf.convert_to_tensor(data[row+1], dtype=tf.float32)

                #make the tensor from size (n_concepts,) to (population, 1, n_concepts)
                self.row_tensor = tf.repeat(tf.reshape(row_tensor, (1, 1, len(row_tensor))),
                                self.pop_size, axis = 0)
                self.row_tensor_test = tf.repeat(tf.reshape(row_tensor_test, (1, 1, len(row_tensor_test))),
                                 self.pop_size, axis = 0)

                self.inference(l, fcm_iterations, verbose)
                f_x += self.fitness_fn(self.row_tensor_test, self.fcm_outputs_tensor)
            
            if i == 0:
                self.f_p = f_x / dataset_length 
                try:
                    self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]] #global best
                    print('initialized global best on row{}'.format(row))
                except:
                    self.g = self.p[tf.math.argmin(input=self.f_p).numpy()] #global best
            
            self.step(f_x / dataset_length)
            
            # self.escape_local_minima(i)
            if self.exploration_decay:
                self.linear_decay(i)
                self.social_to_cognitive(i)

            if self.pso_termination(i):
                break

    def fcm_learning_classification(self, input_data, output_data, transfer_function,l = 1, fcm_iterations = 30, verbose = -1, exploration_decay = True):
      self.l = l
      self.exploration_decay = exploration_decay
      dataset_length = len(input_data)
      self.output_concepts_len = output_data.shape[-1]
      #self.centroids = {}
      self.p = self.x
      for i in range(self.n_iter):
          print('Iter {}'.format(i))
          f_x = tf.zeros((self.pop_size,1)) #error at each iter
          for row in range(dataset_length):
              row_tensor = tf.convert_to_tensor(input_data[row], dtype=tf.float32)
              row_tensor_test = tf.convert_to_tensor(output_data[row], dtype=tf.float32)

              #make the tensor from size (n_concepts,) to (population, 1, n_concepts)
              self.row_tensor = tf.repeat(tf.reshape(row_tensor, (1, 1, len(row_tensor))),
                              self.pop_size, axis = 0)
              self.row_tensor_test = tf.repeat(tf.reshape(row_tensor_test, (1, 1, len(row_tensor_test))),
                                self.pop_size, axis = 0)

              self.inference_class(l= self.l, n_iter = fcm_iterations,verbose = verbose, transfer_function = transfer_function)
              f_x += self.fitness_fn(self.row_tensor_test, self.fcm_outputs_tensor)
          
          if i == 0 and row >= dataset_length - 1:
              self.f_p = f_x / dataset_length  #average the error
              try:
                  self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0]] #global best
                  print('initialized global best on row{}'.format(row) )
              except:
                try:
                  self.g = self.p[tf.math.argmin(input=self.f_p).numpy()] #global best
                  print('initialized global best on row{}'.format(row))
                except:
                  self.g = self.p[tf.math.argmin(input=self.f_p).numpy()[0][0]]
                  print('initialized global best on row{}'.format(row))
          
          self.step(f_x / dataset_length)
          
          # self.escape_local_minima(i)
          if self.exploration_decay:
              self.linear_decay(i)
              self.social_to_cognitive(i)

          if self.pso_termination(i):
              break
      
      if self.output_concepts_len == 1:
        print('Computing centroids for classification with one output...\n')
        self.compute_centroids(transfer_function,input_data,output_data, fcm_iterations)

    def compute_centroids(self,transfer_function, input_data, labels, iterations, l =1):
      classes = np.unique(labels)
      dic_classes = {}
      results = []
      for i in classes:
        dic_classes[i] = []
      for i in range(len(input_data)):
        map = fcm(input_data[i][None, :], self.g, iterations)
        result = map.inference(lambda x: sigmoid(x,self.l), classification = True, output_concepts=labels.shape[-1], verbose = -1).numpy()[:,-labels.shape[-1]:][0]
        dic_classes[labels[i][0]].append(result)
        results.append(result)
      self.centroids = {}
      self.km_centroids = {}
      kmeans = KMeans(n_clusters = len(classes))
      predictions = kmeans.fit_predict(np.array(results))
      km_array = np.sort(kmeans.cluster_centers_[:,0])
      for i,value in enumerate(classes):
        self.centroids[value] = np.mean(dic_classes[value])
        self.km_centroids[value] = km_array[i]
      self.thresholds = [0.0]
      for i in range(len(classes)-1):
        current_mean = np.mean(dic_classes[classes[i]])
        current_std = np.std(dic_classes[classes[i]])
        next_mean = np.mean(dic_classes[classes[i+1]])
        next_std = np.std(dic_classes[classes[i+1]])
        self.thresholds.append( np.mean( (current_mean + current_std, next_mean - next_std ) ) )
      self.thresholds.append(1.0)

    
    def inference(self,l, n_iter, verbose):
        self.fcm_outputs = []
        for i in range(len(self.row_tensor)):
            result = fcm(self.row_tensor[i], self.x[i], n_iterations=n_iter).inference(lambda x: sigmoid(x,l), verbose = verbose, classification = False)
            self.fcm_outputs.append(result)
        self.fcm_outputs_tensor = tf.concat([result[:,None] for result in self.fcm_outputs],0)

    def inference_class(self,l , n_iter, verbose, transfer_function):
        self.fcm_outputs = []
        for i in range(len(self.row_tensor)):
            result = fcm(self.row_tensor[i], self.x[i], n_iterations=n_iter).inference(lambda x: sigmoid(x,l), verbose = verbose, classification = True, output_concepts = self.output_concepts_len)
            self.fcm_outputs.append(result)
        self.fcm_outputs_tensor = tf.concat([result[:,None] for result in self.fcm_outputs],0)


    def pso_termination(self, iteration, steps_ratio = 0.05):
        '''
        Termination Conditions for pso:
        Stop PSO if min of best fit_history doesn't change 
        '''
        wait_steps = int(steps_ratio * self.n_iter)
        min_fit_index = np.argmin(self.min_history)
        if iteration-min_fit_index > wait_steps:
            print('PSO termination at step {}, best fit at step {}'.format(iteration, min_fit_index))
            return True
        else:
            return False

    def linear_decay(self, iteration, target = 0.1):
        initial_b = 0.90
        if iteration > 3 and self.b >= target:
            self.b -= initial_b / self.n_iter
        elif iteration <= 3:
            self.b = initial_b

    def social_to_cognitive(self, iteration):
        c1 = 1.5
        c2 = 0.5
        if iteration > 3:
            self.c1 -= (c1 / self.n_iter)
            self.c2 += (c1 / self.n_iter)
        elif iteration <= 3:
            self.c1 = c1
            self.c2 = c2
    
    def escape_local_minima(self, iteration, particles_ratio = 0.4):
        check_points = [int(0.6*self.n_iter), int(0.8*self.n_iter)]
        best_fit = np.argmin(self.fit_history)
        len_fit_history = len(self.fit_history)
        if len_fit_history - best_fit > self.n_iter // 2 and iteration in check_points:
            random_particles = np.random.randint(self.pop_size, size = int(particles_ratio * self.pop_size))
            new_swarm_size = list(self.dim)
            new_swarm_size.insert(0, int(particles_ratio * self.pop_size))
            random_v = np.random.uniform(self.x_min, self.x_max, new_swarm_size)
            v = self.v.numpy()
            v[random_particles] = random_v
            self.v = tf.convert_to_tensor(v, dtype = tf.float32)
            self.c1, self.c2, self.b = 0.7, 0.3, 0.8
            self.fix_diag()
            if len(self.input_concepts_indexes) > 0:
                for i in self.input_concepts_indexes:
                    self.specify_input_concept(i)
            if len(self.output_concepts_indexes) > 0:
                for i in self.output_concepts_indexes:
                    self.specify_output_concept(i)
        elif len_fit_history - best_fit <= self.n_iter // 2:
            self.c1, self.c2 = 0.5, 0.5


def objective_function(X, Y):
    return tf.losses.MSE(X, Y[:,:,-X.shape[-1]:])
def fitness_function():
    def f(X, Y):
        return objective_function(X, Y)
    return f




# import numpy as np
# from copy import deepcopy as dc

# class PSO:
#     def __init__(self, fitness_fn, dim,
#                  pop_size=100,
#                  n_iter=300,
#                  b=0.7,
#                  c1=0.4,
#                  c2=0.6,
#                  x_min=-1,
#                  x_max=1,
#                  x_offset=0.7):
#         self.fitness_fn = fitness_fn
#         self.pop_size = pop_size
#         self.dim = dim
#         self.n_iter = n_iter
#         self.b = b
#         self.c1 = c1
#         self.c2 = c2
#         self.x_min = x_min
#         self.x_max = x_max
#         self.offset = x_offset

#         self.x = self.build_swarm()
#         self.fit_history = []
#         self.min_history = []
#         self.v = self.start_velocities()
#         self.fix_diag()
#         self.input_concepts_indexes = []
#         self.output_concepts_indexes = []

#     def build_swarm(self):
#         swarm_size = [self.pop_size] + list(self.dim)
#         return np.random.uniform(self.x_min + self.offset, self.x_max - self.offset, swarm_size)

#     def start_velocities(self):
#         swarm_size = [self.pop_size] + list(self.dim)
#         return np.random.uniform(self.x_min + self.offset, self.x_max - self.offset, swarm_size)

#     def fix_diag(self):
#         for i in range(self.pop_size):
#             np.fill_diagonal(self.x[i], 0)
#             np.fill_diagonal(self.v[i], 0)

#     def set_initialization(self, weight_matrix):
#         assert weight_matrix.shape == self.dim
#         self.x = np.tile(weight_matrix, (self.pop_size, 1, 1))
#         self.fix_diag()

#     def specify_input_concept(self, column_index):
#         if column_index not in self.input_concepts_indexes:
#             self.input_concepts_indexes.append(column_index)
#         self.x[:, :, column_index] = 0
#         self.v[:, :, column_index] = 0
#         print('Concept of column index {}, was set as input concept.'.format(column_index))

#     def specify_output_concept(self, row_index):
#         if row_index not in self.output_concepts_indexes:
#             self.output_concepts_indexes.append(row_index)
#         self.x[:, row_index, :] = 0
#         self.v[:, row_index, :] = 0
#         print('Concept of row index {}, was set as output concept.'.format(row_index))

#     def get_randoms(self):
#         return np.random.uniform(0, 1, (2, *self.dim))

#     def update_p_best(self, f_x):
#         self.fit_history.append(np.mean(f_x))
#         self.min_history.append(np.min(f_x))
#         print('minimum error {}'.format(np.min(f_x)))
#         better_fit_mask = f_x < self.f_p
#         self.p = np.where(better_fit_mask[:, None, None], self.x, self.p)
#         self.f_p = np.where(better_fit_mask, f_x, self.f_p)

#     def update_g_best(self):
#         self.g = self.p[np.argmin(self.f_p)]

#     def step(self, f_x):
#         r1, r2 = self.get_randoms()
#         self.v = (
#             self.b * self.v
#             + self.c1 * r1 * (self.p - self.x)
#             + self.c2 * r2 * (self.g - self.x)
#         )
#         self.x = np.clip(self.x + self.v, self.x_min, self.x_max)
#         self.update_p_best(f_x)
#         self.update_g_best()
#         print(self.g)

#     def fcm_learning(self, data, l=1, fcm_iterations=20, verbose=-1, exploration_decay=True):
#         self.exploration_decay = exploration_decay
#         dataset_length = len(data)
#         test_range = [i + 1 for i in range(dataset_length - 1)]
#         self.p = self.x
#         for i in range(self.n_iter):
#             print('Iter {}'.format(i))
#             f_x = np.zeros((self.pop_size, 1))
#             for row in range(dataset_length - 1):
#                 testing_row = test_range[row]
#                 row_tensor = data[row]
#                 row_tensor_test = data[testing_row]

#                 self.row_tensor = np.tile(row_tensor, (self.pop_size, 1, 1))
#                 self.row_tensor_test = np.tile(row_tensor_test, (self.pop_size, 1, 1))

#                 self.inference(l, fcm_iterations, verbose)
#                 f_x += self.fitness_fn(self.row_tensor_test, self.fcm_outputs_tensor)

#             if i == 0 and row >= dataset_length - 1:
#                 self.f_p = f_x / dataset_length
#                 self.g = self.p[np.argmin(self.f_p)]

#             self.step(f_x / dataset_length)

#             if self.exploration_decay:
#                 self.linear_decay(i)
#                 self.social_to_cognitive(i)

#             if self.pso_termination(i):
#                 break

#     def inference(self, l, n_iter, verbose):
#         self.fcm_outputs = []
#         for i in range(self.pop_size):
#             result = fcm(self.row_tensor[i], self.x[i], n_iterations=n_iter).inference(lambda x: sigmoid(x, l), verbose=verbose, classification=False)
#             self.fcm_outputs.append(result)
#         self.fcm_outputs_tensor = np.stack(self.fcm_outputs, axis=0)

#     def pso_termination(self, iteration, steps_ratio=0.05):
#         wait_steps = int(steps_ratio * self.n_iter)
#         min_fit_index = np.argmin(self.min_history)
#         if iteration - min_fit_index > wait_steps:
#             print('PSO termination at step {}, best fit at step {}'.format(iteration, min_fit_index))
#             return True
#         else:
#             return False

#     def linear_decay(self, iteration, target=0.1):
#         initial_b = 0.90
#         if iteration > 3 and self.b >= target:
#             self.b -= initial_b / self.n_iter
#         elif iteration <= 3:
#             self.b = initial_b

#     def social_to_cognitive(self, iteration):
#         c1 = 1.5
#         c2 = 0.5
#         if iteration > 3:
#             self.c1 -= (c1 / self.n_iter)
#             self.c2 += (c1 / self.n_iter)
#         elif iteration <= 3:
#             self.c1 = c1
#             self.c2 = c2

#     def escape_local_minima(self, iteration, particles_ratio=0.4):
#         check_points = [int(0.6 * self.n_iter), int(0.8 * self.n_iter)]
#         best_fit = np.argmin(self.fit_history)
#         len_fit_history = len(self.fit_history)
#         if len_fit_history - best_fit > self.n_iter // 2 and iteration in check_points:
#             random_particles = np.random.choice(self.pop_size, size=int(particles_ratio * self.pop_size), replace=False)
#             new_particles = np.random.uniform(self.x_min + self.offset, self.x_max - self.offset, (len(random_particles), *self.dim))
#             self.x[random_particles] = new_particles
#             self.fix_diag()
