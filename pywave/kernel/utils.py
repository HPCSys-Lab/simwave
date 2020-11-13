# calculate the list of time values
def calc_time_values(timesteps, dt):
    time_values = np.zeros(timesteps, dtype=np.float32)
    t = 0
    for i in range(timesteps):
        time_values[i] = t
        t += dt
