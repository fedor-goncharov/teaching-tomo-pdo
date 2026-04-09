import numpy as np

def solution_line_projector(image, phi, shift, radius=1.0):
    
    # line is outside of the surrounding square
    if (shift > radius*np.sqrt(2)):
        return 0
    
    # image size
    npixels = image.shape[0]
    dx = 2*radius / npixels # pixel's side length 
    
    # set the geometry of the line
    line_center = np.array([[shift * np.cos(phi)], [shift * np.sin(phi)]])
    direction = np.array([[-np.sin(phi)], [np.cos(phi)]])
    dstep = dx / 2
    
    # set integration borders along the line
    lim_left = -np.sqrt(2*radius-shift**2)
    lim_right = -lim_left
    npoints = 2*np.ceil(2*np.sqrt(2*radius-shift**2)/dstep).astype(int) + 1
    
    # get sampling points along the line
    line_points = line_center +  np.multiply(direction, np.linspace(lim_left, lim_right, npoints))
    
    line_pixels = np.floor((line_points + radius)/dx).astype(int) # integers (i,j) of pixel coordinates
    line_pixels[1, :] = npixels - 1 - line_pixels[1, :]
    
    line_pixels = line_pixels[:, ((line_pixels[0] > -1)*(line_pixels[0] < npixels))
                              *((line_pixels[1] > -1)*(line_pixels[1] < npixels))]
    
    line_image_values = image[line_pixels[1, :], line_pixels[0, :]]
    proj_value = np.sum(line_image_values) * dx

    return proj_value


def radon2d(image, ntheta, nshift, radius=1.0):
    
    # image size
    npixels = image.shape[0]
    dx = 2.0*radius / npixels
    shifts = np.linspace(-radius + dx/2, radius-dx/2, nshift)
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint = False)
    
    # compute projections 
    proj = np.zeros((ntheta, nshift))
    for i_theta in range(ntheta):
        for i_shift in range(nshift):
            proj[i_theta][i_shift] = solution_line_projector(image, theta[i_theta], 
                                                          shifts[i_shift], radius)    
    return proj


def adjradon2d(projections, npixels = 128, dom_rad = 1.0):
    
    # init geometry in the domain
    ntheta = projections.shape[0]  
    nshift = projections.shape[1]
    
    dtheta   = 2 * np.pi / ntheta;
    dshift = 2./ nshift
    array_theta = np.arange(ntheta) * dtheta
    shifts = np.linspace(-1. + dshift/2, 1. - dshift/2, nshift)

    # create 2D grid 
    lin = np.linspace(-dom_rad, dom_rad, npixels)
    XX,YY = np.meshgrid(lin, -lin)
    XX = np.reshape(XX, (1, npixels**2))
    YY = np.reshape(YY, (1, npixels**2))

    adjoint = np.zeros((1, npixels**2))
    grid_points = np.concatenate((XX, YY))

    for i_theta in range(ntheta):

        theta = array_theta[i_theta]
        direction = np.array([np.cos(theta), np.sin(theta)])
        
        new_shifts = np.dot(np.transpose(grid_points), direction)
        add_interpolated = np.interp(new_shifts, shifts, projections[i_theta, :],  left=0., right=0.)
        adjoint += add_interpolated
        
    adjoint = np.reshape(adjoint, (npixels, npixels)) * dtheta;
    
    return adjoint
