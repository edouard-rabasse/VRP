def read_arcs(file_path: str, type: str = "original", number_of_fields: int = None) -> list:
    """
    Read VRP arcs from semicolon-separated file.
    
    Args:
        file_path: Path to arc data file  
        type: Arc type (obsolete parameter)
        number_of_fields: Number of fields to read per arc
        
    Returns:
        List of arc tuples
    """
    arcs = []
    with open(file_path, "r") as file:
        # Read the first line to determine the type
        first_line = file.readline().strip()
        num_fields = len(first_line.split(";"))

        # Seek back to the beginning of the file
        file.seek(0)

        for line in file:
            fields = list(map(int, line.strip().split(";")))
            if number_of_fields is None:
                number_of_fields = num_fields
            if num_fields == 4:
                arcs.append(tuple(fields[:number_of_fields]))
            elif num_fields == 5:
                arcs.append(tuple(fields[:number_of_fields]))
            else:
                raise ValueError("Unsupported number of fields in the file.")

    return arcs


def binarize_arcs(arcs: list, threshold: float = 0, index: int = 3) -> list:
    """
    Convert arc values to binary based on threshold.
    
    Args:
        arcs: List of arc tuples
        threshold: Threshold for binarization  
        index: Field index to apply threshold on
        
    Returns:
        List of arcs with binarized values
    """
    binarized_arcs = []
    for arc in arcs:
        if len(arc) > index:
            binarized_value = 1 if arc[index] > threshold else 0
            binarized_arcs.append(arc[:index] + (binarized_value,))
    return binarized_arcs


def isolate_top_arcs(flagged_arcs: list[tuple], index=4, number=3):
    """
    Met à 0 la valeur à l'index spécifié pour tous les arcs qui ne sont pas
    parmi les 'number' arcs avec les plus grosses valeurs à cet index.

    Args:
        flagged_arcs (list): Liste des arcs marqués.
        index (int): Index du champ à utiliser pour le tri (défaut: 4).
        number (int): Nombre d'arcs top à conserver (défaut: 3).

    Returns:
        list: Liste des arcs avec les valeurs modifiées.
    """
    try:
        if not flagged_arcs:
            return []

        if number <= 0:
            # Si number <= 0, tous les arcs sont mis à 0
            return [arc[:index] + (0,) + arc[index + 1 :] for arc in flagged_arcs]

        if number >= len(flagged_arcs):
            # Si on veut garder plus d'arcs qu'il n'y en a, on retourne tout
            return flagged_arcs.copy()

        # Trier les arcs par valeur décroissante à l'index spécifié
        sorted_arcs = sorted(flagged_arcs, key=lambda x: x[index], reverse=True)

        # Trouver la valeur seuil (valeur du number-ième arc)
        threshold_value = sorted_arcs[number - 1][index]

        # Créer la liste résultante
        result_arcs = []
        top_count = 0

        for arc in flagged_arcs:
            # Garder l'ordre original des arcs
            if arc[index] > threshold_value:
                # Arc clairement dans le top
                result_arcs.append(arc)
                top_count += 1
            elif arc[index] == threshold_value and top_count < number:
                # Arc avec valeur égale au seuil, garder si on n'a pas atteint le quota
                result_arcs.append(arc)
                top_count += 1
            else:
                # Arc pas dans le top, mettre la valeur à 0
                new_arc = arc[:index] + (0,) + arc[index + 1 :]
                result_arcs.append(new_arc)

        return result_arcs

    except Exception as e:
        raise ValueError(f"Failed to isolate top arcs: {e}") from e


def get_arc_name(index: int, suffix: int | str = 1) -> str:
    """
    Generate standardized arc filename.
    
    Args:
        index: Instance index number
        suffix: Configuration suffix
        
    Returns:
        Formatted arc filename string
    """
    return f"Arcs_{index}_{suffix}.txt"


if __name__ == "__main__":
    # Example usage
    arcs = read_arcs("MSH/MSH/results/configuration1/Arcs_1_1.txt")
    print(arcs)

    binarized_arcs = binarize_arcs(arcs, threshold=-1, index=4)
    print(binarized_arcs)

    arc_name = get_arc_name(1, suffix=2)
    print(arc_name)
