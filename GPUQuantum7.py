import pygame
import math
import numpy as np
import pyopencl as cl
import pyopencl.array
import cv2
import warnings

# Suppress the warning about the hash function
warnings.filterwarnings("ignore", category=UserWarning)

pygame.init()

width, height = 1000, 700
screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Interactive Double Slit Experiment (GPU Accelerated)")

BLACK, CYAN, PURPLE, WHITE, RED, GREEN = (0, 0, 0), (0, 255, 255), (139, 0, 255), (255, 255, 255), (255, 0, 0), (0, 255, 0)

font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

barrier_x, slit_height, slit_gap = int(width * 0.7), 30, int(height * 0.2)
slit1_y, slit2_y = int(height * 0.4), int(height * 0.6)

wave_speed = 0.1
wavelength = 40
k = 2 * math.pi / wavelength
wave_amplitude = 5.0

# Use GPU 0
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

kernel_code = """
__kernel void calculate_wave_field(__global float* wave_field,
                                   int width, int height, int barrier_x,
                                   float k, float time, int slit1_y, int slit2_y, int slit_height,
                                   float wave_amplitude)
{
    int x = get_global_id(0) + barrier_x;
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    float dx1 = (float)(x - barrier_x);
    float dy1 = (float)(y - slit1_y);
    float r1 = sqrt(dx1 * dx1 + dy1 * dy1);
    
    float dx2 = (float)(x - barrier_x);
    float dy2 = (float)(y - slit2_y);
    float r2 = sqrt(dx2 * dx2 + dy2 * dy2);
    
    float amplitude = wave_amplitude * (sin(k * r1 - time) / sqrt(r1 + 1.0f) + sin(k * r2 - time) / sqrt(r2 + 1.0f));
    
    wave_field[y * width + x] = amplitude;
}
"""

program = cl.Program(ctx, kernel_code).build()
wave_field = pyopencl.array.zeros(queue, (height, width), dtype=np.float32)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

max_particles = 1000
particles = np.zeros((max_particles, 3), dtype=np.float32)  # x, y, stuck
particle_count = 0
emission_rate, particle_speed = 20, 5

camera_button = pygame.Rect(20, height - 70, 200, 50)

use_camera, paused = True, False

def calculate_wave_field(time):
    global wave_field
    program.calculate_wave_field(queue, (width - barrier_x, height), None,
                                 wave_field.data, np.int32(width), np.int32(height), np.int32(barrier_x),
                                 np.float32(k), np.float32(time),
                                 np.int32(slit1_y), np.int32(slit2_y), np.int32(slit_height),
                                 np.float32(wave_amplitude))

def draw_interference(surface):
    field_data = wave_field.get()
    pixels = pygame.surfarray.pixels3d(surface)
    normalized_data = np.clip(field_data[:, barrier_x:] / wave_amplitude, -1, 1)
    pixels[barrier_x:, :, 0] = np.clip(255 * (normalized_data > 0) * normalized_data, 0, 255).T
    pixels[barrier_x:, :, 1] = np.clip(255 * np.abs(normalized_data), 0, 255).T
    pixels[barrier_x:, :, 2] = np.clip(255 * (normalized_data < 0) * -normalized_data, 0, 255).T
    del pixels

def draw_barrier(surface):
    pygame.draw.rect(surface, PURPLE, (barrier_x - 5, 0, 10, height))
    pygame.draw.rect(surface, BLACK, (barrier_x - 5, slit1_y - slit_height//2, 10, slit_height))
    pygame.draw.rect(surface, BLACK, (barrier_x - 5, slit2_y - slit_height//2, 10, slit_height))

def detect_face():
    if not use_camera:
        return False
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return len(faces) > 0
    return False

def emit_particle():
    global particle_count
    if particle_count < max_particles:
        particles[particle_count] = [0, np.random.randint(0, height), 0]
        particle_count += 1

def update_particles():
    global particles, particle_count
    # Move particles from left to the barrier
    mask = (particles[:particle_count, 0] < barrier_x - 5) & (particles[:particle_count, 2] == 0)
    particles[:particle_count, 0][mask] += particle_speed

    # Handle particle interaction with the barrier
    mask = (barrier_x - 5 <= particles[:particle_count, 0]) & (particles[:particle_count, 0] <= barrier_x + 5) & (particles[:particle_count, 2] == 0)
    slit_mask = ((slit1_y - slit_height//2 <= particles[:particle_count, 1]) & (particles[:particle_count, 1] <= slit1_y + slit_height//2)) | \
                ((slit2_y - slit_height//2 <= particles[:particle_count, 1]) & (particles[:particle_count, 1] <= slit2_y + slit_height//2))
    
    if face_detected:  # Particle view: particles pass through slits
        particles[:particle_count, 0][mask & slit_mask] = barrier_x + 6
        particles[:particle_count, 2][mask & ~slit_mask] = 1
    else:  # Wave view: particles stop at the barrier
        particles[:particle_count, 2][mask] = 1
        particles[:particle_count, 0][mask] = barrier_x - 6

    # Move particles to the right of the barrier (only in particle view)
    if face_detected:
        mask = (particles[:particle_count, 0] > barrier_x + 5) & (particles[:particle_count, 0] < width - 2) & (particles[:particle_count, 2] == 0)
        particles[:particle_count, 0][mask] += particle_speed

        particles[:particle_count, 2][(particles[:particle_count, 0] >= width - 2) & (particles[:particle_count, 2] == 0)] = 1
        particles[:particle_count, 0][(particles[:particle_count, 0] >= width - 2) & (particles[:particle_count, 2] == 0)] = width - 2

def draw_particles(surface):
    for i in range(particle_count):
        if particles[i, 0] <= barrier_x or face_detected:
            pygame.draw.circle(surface, CYAN, (int(particles[i, 0]), int(particles[i, 1])), 2)

def draw_ui(surface):
    pygame.draw.rect(surface, GREEN if use_camera else RED, camera_button)
    surface.blit(font.render("Toggle Camera", True, BLACK), (camera_button.x + 10, camera_button.y + 10))
    surface.blit(small_font.render("Press SPACE to pause/resume. Click button to toggle camera.", True, WHITE), (20, 20))
    if paused:
        surface.blit(font.render("PAUSED", True, RED), (width // 2 - 50, 20))
    surface.blit(font.render("Particle View" if face_detected else "Wave View", True, WHITE), (width - 200, 20))

running = True
clock = pygame.time.Clock()
frame_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if camera_button.collidepoint(event.pos):
                use_camera = not use_camera
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:
        screen.fill(BLACK)

        face_detected = detect_face()

        if frame_count % (60 // emission_rate) == 0:
            emit_particle()

        update_particles()
        
        current_time = frame_count * wave_speed
        
        calculate_wave_field(current_time)
        if not face_detected:
            draw_interference(screen)
        
        draw_particles(screen)
        draw_barrier(screen)
        draw_ui(screen)

        pygame.display.flip()
        frame_count += 1

    clock.tick(60)

pygame.quit()
cap.release()
cv2.destroyAllWindows()