import numpy as np
from qutip import Bloch, Qobj, sigmax, sigmay, sigmaz, qeye
import imageio
import os

# Function to compute Bloch vector from density matrix
def BlochVector(rho):
    x = np.real((rho * sigmax()).tr())
    y = np.real((rho * sigmay()).tr())
    z = np.real((rho * sigmaz()).tr())
    return np.array([x, y, z])

# Function to apply amplitude damping
def amplitude_damping(rho, gamma):
    K0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = Qobj([[0, np.sqrt(gamma)], [0, 0]])
    return K0 * rho * K0.dag() + K1 * rho * K1.dag()

# Function to apply phase damping
def phase_damping(rho, gamma):
    K0 = Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = Qobj([[0, 0], [0, np.sqrt(gamma)]])
    return K0 * rho * K0.dag() + K1 * rho * K1.dag()

# Function to apply depolarizing channel
def depolarizing_channel(rho, gamma):
    K0 = np.sqrt(1 - 3 * gamma / 4) * qeye(2)
    K1 = np.sqrt(gamma / 4) * sigmax()
    K2 = np.sqrt(gamma / 4) * sigmay()
    K3 = np.sqrt(gamma / 4) * sigmaz()
    return K0 * rho * K0.dag() + K1 * rho * K1.dag() + K2 * rho * K2.dag() + K3 * rho * K3.dag()

# Function to generate a density matrix from spherical coordinates
def densitymatrix_fromangles(theta, phi):
    density_matrix = 0.5 * (
        qeye(2)
        + np.sin(theta) * np.cos(phi) * sigmax()
        + np.sin(theta) * np.sin(phi) * sigmay()
        + np.cos(theta) * sigmaz()
    )
    return density_matrix

# Function to compute Bloch vector after applying damping channel
def BlochVectorafterDamping(theta, phi, gamma, channel_type):
    rho = densitymatrix_fromangles(theta, phi)
    if channel_type == "amplitude damping":
        dampedrho = amplitude_damping(rho, gamma)
    elif channel_type == "phase damping":
        dampedrho = phase_damping(rho, gamma)
    elif channel_type == "depolarization":
        dampedrho = depolarizing_channel(rho, gamma)
    return BlochVector(dampedrho)

# Function to smoothly move the vector towards the north pole
def contractedBlochVector(theta, phi, gamma, channel_type):
    blochvec = BlochVectorafterDamping(theta, phi, gamma, channel_type)
    scale = 1 - gamma
    if channel_type == "amplitude damping":
        return np.array([blochvec[0] * scale, blochvec[1] * scale, 1 - scale * (1 - blochvec[2])])
    elif channel_type == "phase damping":
        return np.array([blochvec[0] * scale, blochvec[1] * scale, blochvec[2]])
    elif channel_type == "depolarization":
        return np.array([blochvec[0] * scale, blochvec[1] * scale, scale * blochvec[2]])

# Function to generate Bloch sphere animation
def bloch_sphere_animation(output_file, theta, phi, channel_type):
    gamma_values = np.linspace(0, 1, 100)  # Smooth transition from 0 to 1
    frames = []
    bloch = Bloch()

    # Generate animation frames
    for gamma in gamma_values:
        bloch.clear()  # Clear previous vectors
        bloch.vector_color = ['r']
        
        # Add the vector at each stage
        contracted_vec = contractedBlochVector(theta, phi, gamma, channel_type)
        bloch.add_vectors(contracted_vec)
        
        # Save each frame
        frame_path = f"frame_{gamma:.2f}.png"
        bloch.save(frame_path)
        frames.append(imageio.imread(frame_path))
        
        # Clean up the saved frame to save disk space
        os.remove(frame_path)
    
    # Save animation as a GIF
    imageio.mimsave(output_file, frames, duration=0.1)
    print(f"Animation saved as {output_file}")

# Generate the Bloch sphere animation
bloch_sphere_animation("BlochSphereAnimation.gif", theta=np.pi/8, phi=np.pi/8, channel_type="depolarization")
