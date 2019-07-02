class Client:
    def __init__(self, client_id, task, queue, table):
        table.identify(client_id)
        queue.identify(client_id)
        task(client_id, queue, table)
