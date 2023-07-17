import tkinter
import customtkinter
from PIL import ImageTk, Image
import mysql.connector
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
from sklearn.neural_network import MLPClassifier


customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("green")

# Connect to the MySQL database
cnx = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='register_user'
)

# Create a cursor object to interact with the database
cursor = cnx.cursor()

app = customtkinter.CTk()
app.geometry("600x440")
app.title('Registration')
# Create an empty dictionary to store keystroke dynamics
keystroke_data = {}

def capture_timing_data(event):
    key = event.char
    time_stamp = time.time()

    if key not in keystroke_data:
        keystroke_data[key] = {
            'press_times': [],
            'release_times': [],
            'hold_times': [],
            'keydown_keydown_times': [],
            'keyup_keydown_times': []
        }
    else:
        key_data = keystroke_data[key]
        if event.keysym == 'KeyPress':
            key_data['press_times'].append(time_stamp)
        elif event.keysym == 'KeyRelease':
            key_data['release_times'].append(time_stamp)
            if len(key_data['press_times']) > 1:
                keydown_keydown_time = key_data['press_times'][-1] - key_data['press_times'][-2]
                key_data['keydown_keydown_times'].append(keydown_keydown_time)
            if len(key_data['release_times']) > 0 and len(key_data['press_times']) > 0:
                keyup_keydown_time = key_data['press_times'][-1] - key_data['release_times'][-1]
                key_data['keyup_keydown_times'].append(keyup_keydown_time)
        if len(key_data['press_times']) > 0 and len(key_data['release_times']) > 0:
            hold_time = key_data['release_times'][-1] - key_data['press_times'][-1]
            key_data['hold_times'].append(hold_time)


def button_function():
    app.destroy()
    w = customtkinter.CTk()
    w.geometry("1280x720")
    w.title('Welcome')
    l1 = customtkinter.CTkLabel(master=w, text="Keystrokes Dynamics Captured", font=('Century Gothic', 60))
    l1.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)
    w.mainloop()

    # Preprocess the captured keystroke dynamics data
    hold_times = []
    keydown_keydown_times = []
    keyup_keydown_times = []

    for key, key_data in keystroke_data.items():
        if len(key_data['hold_times']) > 0:
            hold_times.extend(key_data['hold_times'])
        if len(key_data['keydown_keydown_times']) > 0:
            keydown_keydown_times.extend(key_data['keydown_keydown_times'])
        if len(key_data['keyup_keydown_times']) > 0:
            keyup_keydown_times.extend(key_data['keyup_keydown_times'])

    # Preprocess the data
    scaler = StandardScaler()

    # Convert lists to numpy arrays
    hold_times_arr = np.array(hold_times).reshape(-1, 1)
    keydown_keydown_times_arr = np.array(keydown_keydown_times).reshape(-1, 1)
    keyup_keydown_times_arr = np.array(keyup_keydown_times).reshape(-1, 1)

    # Scale the data
    scaled_hold_times = scaler.fit_transform(hold_times_arr)
    scaled_keydown_keydown_times = scaler.fit_transform(keydown_keydown_times_arr)
    scaled_keyup_keydown_times = scaler.fit_transform(keyup_keydown_times_arr)

    # Load the pre-trained machine learning model
    model = pickle.load("model_keystroke.pkl")

    # Create a multilayer perceptron classifier
    net = MLPClassifier(random_state=42, hidden_layer_sizes=(84,), max_iter=600, activation='relu',
                        learning_rate='invscaling', solver='adam')

    # Prepare the data for training
    data = np.concatenate((scaled_hold_times, scaled_keydown_keydown_times, scaled_keyup_keydown_times), axis=1)

    # Reshape the data to match the input shape expected by the model
    data = np.reshape(data, (1, -1))

    # train model
    train = model.train(data)

img1 = ImageTk.PhotoImage(Image.open("pattern.png"))
l1 = customtkinter.CTkLabel(master=app, image=img1)
l1.pack()

frame = customtkinter.CTkFrame(master=l1, width=320, height=360, corner_radius=15)
frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

l2 = customtkinter.CTkLabel(master=frame, text="Registration", font=('Century Gothic', 20))
l2.place(x=50, y=45)

entry1 = customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Username')
entry1.place(x=50, y=110)

entry2 = customtkinter.CTkEntry(master=frame, width=220, placeholder_text='Password', show="*")
entry2.place(x=50, y=165)

l3 = customtkinter.CTkLabel(master=frame, text="Forget password?", font=('Century Gothic', 12))
l3.place(x=155, y=195)


def insert_user():
    username = entry1.get()
    password = entry2.get()

    # Insert user data into the 'users' table
    insert_query = '''
        INSERT INTO users (username, password)
        VALUES (%s, %s)
    '''
    cursor.execute(insert_query, (username, password))

    cnx.commit()

    # Call the button_function() after successfully inserting user details
    button_function()


button1 = customtkinter.CTkButton(master=frame, width=220, text="Sign Up", command=insert_user, corner_radius=6)
button1.place(x=50, y=240)

img2 = customtkinter.CTkImage(Image.open("Google__G__Logo.svg.webp").resize((20, 20), Image.BILINEAR))
img3 = customtkinter.CTkImage(Image.open("124010.png").resize((20, 20), Image.BILINEAR))
button2 = customtkinter.CTkButton(master=frame, image=img2, text="Google", width=100, height=20, compound="left",
                                  fg_color='white', text_color='black', hover_color='#AFAFAF')
button2.place(x=50, y=290)

button3 = customtkinter.CTkButton(master=frame, image=img3, text="Facebook", width=100, height=20, compound="left",
                                  fg_color='white', text_color='black', hover_color='#AFAFAF')
button3.place(x=170, y=290)

app.mainloop()

# Close the database connection
cursor.close()
cnx.close()

