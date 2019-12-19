from bfs import bfs_queue

def statistics(graph, nodes, level=6):
    from texttable import Texttable
    from tqdm import tqdm
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    results = {}
    stat = [['Node']]
    stat[0].extend([f'{i}-hop' for i in range(1,level+1)])
    for node in tqdm(nodes):
        res, nodes = bfs_queue(graph, node, level)
        row = [node]
        for hop in range(1, level+1):
            if len(res) < hop+1:
                row.append(0)
            else:
                row.append(len(res[hop]))
        stat.append(row)
        results[node] = (res, nodes)
    table.add_rows(stat)
    print(table.draw())
    return results