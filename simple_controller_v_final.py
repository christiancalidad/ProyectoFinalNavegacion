import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore")

from controller import Display, Keyboard, Robot, Camera, Radar
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv
from time import sleep
import tensorflow as tf
import joblib
from PIL import Image

def get_image(camera):
    #Obtener la imagen cruda de la camara
    raw_image = camera.getImage()
    #Se convierte la imagen cruda a un arreglo numpy
    image = np.frombuffer(raw_image, np.uint8)
    #Se crea una imagen PIL desde el arreglo numpy
    image_pil = Image.frombytes("RGBA", (camera.getWidth(), camera.getHeight()), image)
    #Se convierte la imagen PIL a un arreglo numpy
    image_np = np.array(image_pil)
    #Retornar la imagen como arreglo numpy
    return image_np


manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 20

def predict_steering_angle(model, image):
    #se normaliza la imagen a valor entre 0 y 1
    image = image.astype(np.float32) /255
    # Se realiza la predicción con el modelo
    prediction = model.predict(np.expand_dims(image, axis=0))
    return prediction
 

def set_speed(kmh):
    global speed

def set_steering_angle(wheel_angle):
    global angle, steering_angle
    steering_angle = wheel_angle
    angle = wheel_angle

def change_steer_angle(inc):
    global manual_steering
    new_manual_steering = manual_steering + inc
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    if manual_steering == 0:
        pass
    else:
        turn = "left" if steering_angle < 0 else "right"

def draw_speedometer(display, speed):
    display_width = display.getWidth()
    display_height = display.getHeight()
   
    # Se le pone el fondo al display
    display.setColor(0xFFFFFF)
    display.fillRectangle(0, 0, display_width, display_height)
    display.setFont('Arial', 16, True)
    # se escribe la velocidad en el display
    display.setColor(0x000000)
    display.drawText(f'{speed:.2f} km/h',50,50)

def main():
    global speed, angle

    #Crear una instancia del robot
    robot = Car()
    #Crear una instancia del conductor
    driver = Driver()

    #Obtener el paso de tiempo del mundo actual
    timestep = int(robot.getBasicTimeStep())

    #Crear una instancia de la cámara
    camera = robot.getDevice("camera")
    #Habilitar la cámara con el paso de tiempo
    camera.enable(timestep)
    
    #Crear una instancia del radar
    radar = robot.getDevice("radar")
    #Habilitar el radar con el paso de tiempo
    radar.enable(timestep)
    # se setea el display para poner la velocidad
    display = robot.getDisplay('display')
     
    
    #Definir umbrales y variables de control
    threshold_distance = 7      #Distancia umbral para detección de obstáculos
    reduce_speed = False        #Indicador para reducir la velocidad
    
    #Cargar el escalador
    scaler = joblib.load('scaler.pkl')
    #Cargar el modelo de aprendizaje profundo
    model = tf.keras.models.load_model('model.h5', compile=False)


    while robot.step() != -1:
        draw_speedometer(display, speed)
        targets = radar.getTargets()
        should_stop = False
        #Evaluar cada objetivo detectado por el radar
        for target in targets:
            #Obtener la distancia del objetivo
            distance = target.distance
            #Verificar si el objetivo está dentro del umbral
            if distance <= threshold_distance and distance > 1:
                should_stop = True   #Indicar que se debe detener el vehículo
                print(f"Target detectado a distancia: {distance} metros")
                break  #Salir del bucle al encontrar un objetivo cercano
                
       #Reducir la velocidad o detener el vehículo según los indicadores
        if should_stop:
            reduce_speed = True
            
        else:
            if reduce_speed:
                speed -= 1       #Reducir la velocidad gradualmente
                if speed <= 0:
                    speed = 0              #Asegurarse de que la velocidad no sea negativa
                    reduce_speed = False
            else:
                if speed < 20:
                    speed += 1 
        
        #print(speed)
        image = get_image(camera)
        # Obtener la predicción con el modelo
        predicted_angle = predict_steering_angle(model, image)
        # Calcular el ángulo de giro a partir de la predicción
        predicted_angle = scaler.inverse_transform(predicted_angle)
        angle = float(predicted_angle[0])
        #Establecer el ángulo de dirección y la velocidad de crucero del conductor
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()
