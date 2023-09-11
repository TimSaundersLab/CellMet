


def cell_analysis():
    # Calculate cell lengths and curvature
    real_dist, short_dist, curv_ind = ZManalysis.calculate_lengths_curvature(data_,
                                                                             columns=['X.micron', 'Y.micron',
                                                                                      "Z.micron"])

    # Calculate volume
    c_volume = np.count_nonzero(img_cell) * voxel_size

    # Measure cell plane metrics
    (aniso, orientation_x, orientation_y,
     major, minor, area, perimeter) = ZManalysis.cell_plane_measure(img_cell, pixel_size)


def edge_analysis():
    # Calculate edge lengths and curvature
    real_dist, short_dist, curv_ind = ZManalysis.calculate_lengths_curvature(df,
                                                                             columns=['x', 'y', "z"])

    for i in range(len(z0)):
        cell_pos_plane = cell_pos[cell_pos['z'] == z0[i]]
        if len(cell_pos_plane) != 0:
            x_cell.append((x0[i] - cell_pos_plane['x']).to_numpy()[0])
            y_cell.append((y0[i] - cell_pos_plane['y']).to_numpy()[0])
            z_cell.append(z0[i])

            angle_rot = csutils.get_angle(
                (x_cell[0] * self.pixel_size['x_size'], y_cell[0] * self.pixel_size['y_size']),
                (0, 0),
                (
                    x_cell[-1] * self.pixel_size['x_size'],
                    y_cell[-1] * self.pixel_size['y_size']))
        else:
            angle_rot = np.nan

    cell_df.loc[cell_df[cell_df['id_im'].isin(edge_df.groupby('id_im_1').mean().index)].index,
        'angle_rot_mean'] = edge_df.groupby('id_im_1').mean()['angle_rot'].to_numpy()
        cell_df.loc[cell_df[cell_df['id_im'].isin(edge_df.groupby('id_im_1').mean().index)].index,
        'angle_rot_std'] = edge_df.groupby('id_im_1').std()['angle_rot'].to_numpy()
        cell_df.to_csv(os.path.join(path, "cell_df.csv"))


def face_analysis():
    df['length_um'] = np.sqrt((df['x'] - df['x1']) ** 2.0 + (df['y'] - df['y1']) ** 2.0) * pixel_size[
        'x_size']
    df['angle'] = (np.arctan2((df['y1'] - df['y_mid']).to_numpy(),
                              (df['x1'] - df['x_mid']).to_numpy()) * 180 / np.pi)