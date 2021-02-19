from classes import Epochs
from utils.artifact_remove import artifact_remove
import numpy as np
import mne


# Recebe um arquivo de gravação de eeg e seus eventos, e separa em matrizes tridimensionais de classes: [M, N, C]
# Sendo M o número de eletrodos, N o número de amostras, e C o número de classes no dicionário
def epoch_raw_data(raw, events, e_dict, t_start, t_end, ica_start, ica_end) -> dict:

    # Guarda a quantidade de canais e calcula o numero de amostra das epocas
    ch = raw.pick('eeg').info['nchan']
    n_samp = int((t_end - t_start) * raw.info["sfreq"] + 1)

    # Pré aloca um dicionário que será utilizado como retorno da função
    x = dict()

    # Esse laço roda cada uma das trials dentro de um arquivo
    for n, i in enumerate(events[:, 0] / raw.info["sfreq"]):

        if events[n, 2] not in e_dict:
            continue

        # Salva a classe de movimento atual
        class_mov = e_dict[events[n, 2]]

        # Coleta uma amostra de (ica_end - ica_start) segundos para análise
        raw_samp = raw.copy().pick('eeg').crop(tmin=i+ica_start, tmax=i+ica_end)

        # Realiza a remoção de artefatos
        raw_clean, flag = artifact_remove(raw_samp)
        # raw_clean = raw_samp.copy()

        # Salva a epoca
        new_epc = \
            raw_clean.crop(tmin=t_start, tmax=t_end).get_data().reshape(ch, n_samp, 1)

        # Adiciona o sinal atual em sua respectiva classe do dicionário X
        try:
            x[class_mov].add_epoch(new_epc)
        except KeyError:
            x[class_mov] = Epochs(
                x=new_epc,
                classe=class_mov,
                fs=raw.info["sfreq"],
            )

    return x


# Cria um objeto de montagem de acordo com os canais informados
def sort_montage_eog(dataset_ch_names):
    # Carrega o arquivo
    file = np.loadtxt('plotting_1005.txt', dtype={
        'names': ['ch', 'x', 'y', 'z'],
        'formats': ['S6', 'f4', 'f4', 'f4']
    })

    # Cria a variavel que ira guardar o nome dos canais
    ch_nums = len(file)
    all_ch_names = []
    coord = np.zeros([342, 3])

    # Passa pelo arquivo linha por linha
    for ii, jj in enumerate(file):
        # Converte a primeira coluna para string e salva na lista
        all_ch_names.append(file[ii][0].decode('ascii'))

        # Converte as coordenadas para float e guarda na matriz
        for coo in range(3):
            coord[ii, coo] = float(file[ii][coo + 1]) / 10

    # Salva em uma matriz as posições de cada um dos canais rferenciados em 'names'
    ch_coord = coord[np.where([all_ch_names[i] in dataset_ch_names for i in range(ch_nums)])[0]]

    # Salva a posição do eletrodo Nasio para referencia
    nz_pos = coord[np.where([all_ch_names[i] in ['Nz'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo lpa para referencia
    lpa_pos = coord[np.where([all_ch_names[i] in ['LPA'] for i in range(ch_nums)])[0]].reshape(3)
    # Salva a posição do eletrodo rpa para referencia
    rpa_pos = coord[np.where([all_ch_names[i] in ['RPA'] for i in range(ch_nums)])[0]].reshape(3)

    # Cria o dicionario de montagem do layout
    ch_pos = {k: v for k, v in zip(dataset_ch_names, ch_coord)}

    # Cria o objeto de montagem do laout
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=nz_pos,
        lpa=lpa_pos,
        rpa=rpa_pos
    )

    return montage
