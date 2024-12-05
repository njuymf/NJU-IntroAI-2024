from othello_game import OthelloGame

count1 = 0


def get_best_move(game, max_depth=6):
    """
    Given the current game state, this function returns the best move for the AI player using the Alpha-Beta Pruning
    algorithm with a specified maximum search depth.

    Parameters:
        game (OthelloGame): The current game state.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    """
    # _, best_move = minmax_decider(game, max_depth)
    # _, best_move = alphabeta_decider(game, max_depth)
    # _, best_move = alphabeta_decider2(game, max_depth)
    # _, best_move = alphabeta_decider3(game, max_depth)
    _, best_move = mtd_f(game, 0, max_depth)
    return best_move


def minmax_decider(
        game,
        max_depth,
        maximizing_player=True
):
    """
    MinMax Decider algorithm for selecting the best move for the AI player.

    Parameters:
        game (OthelloGame): The current game state.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.
        maximizing_player (bool): True if maximizing player (AI), False if minimizing player (opponent).

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    """

    global count1

    if max_depth == 0 or game.is_game_over():
        return evaluate_game_state(game), None

    valid_moves = game.get_valid_moves()  # 获取所有合法的移动
    count1 += len(valid_moves)
    print(count1)

    if maximizing_player:  # 如果是AI，那么就找最大值
        max_eval = float("-inf")
        best_move = None

        for move in valid_moves:  # 遍历所有合法的移动
            new_game = OthelloGame(player_mode=game.player_mode)  # 创建一个新的游戏
            new_game.board = [row[:] for row in game.board]  # 复制棋盘
            new_game.current_player = game.current_player  # 复制当前玩家
            new_game.make_move(*move)  # 在新游戏中执行这个移动

            eval, _ = minmax_decider(new_game, max_depth - 1, False)  # 递归调用，找最小值

            if eval > max_eval:  # 找最大值
                max_eval = eval  # 更新最大值
                best_move = move  # 更新最佳移动

        return max_eval, best_move  # 返回最大值和最佳移动
    else:  # 如果是对手，那么就找最小值
        min_eval = float("inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = minmax_decider(new_game, max_depth - 1, True)

            if eval < min_eval:
                min_eval = eval
                best_move = move

        return min_eval, best_move


def alphabeta_decider(
        game,
        max_depth,
        maximizing_player=True,
        alpha=float("-inf"),
        beta=float("inf")
):
    """
    Alpha-Beta Pruning Enhanced MinMax Decider algorithm for selecting the best move for the AI player.

    参数:
        game (OthelloGame): 当前的游戏状态。
        max_depth (int): Alpha-Beta算法的最大搜索深度。
        maximizing_player (bool): 如果是最大化玩家（AI），则为True；如果是最小化玩家（对手），则为False。
        alpha (float): 当前Alpha值，初始为负无穷。
        beta (float): 当前Beta值，初始为正无穷。

    返回:
        tuple: 包含最佳移动的评估值和对应的移动（row, col）。
    """

    global count1

    # 基线条件：达到最大深度或游戏结束
    if max_depth == 0 or game.is_game_over():
        # return evaluate_game_state(game), None
        return improved_evaluate_game_state(game), None

    valid_moves = game.get_valid_moves()  # 获取所有合法的移动
    # count1 += len(valid_moves)
    # print(count1)

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in valid_moves:
            # 复制当前游戏状态
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            # 递归调用，切换为最小化玩家，并传递当前的alpha和beta值
            eval, _ = alphabeta_decider(new_game, max_depth - 1, False, alpha, beta)

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)  # 更新Alpha值
            if alpha >= beta:
                break  # Beta剪枝

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in valid_moves:
            # 复制当前游戏状态
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            # 递归调用，切换为最大化玩家，并传递当前的alpha和beta值
            eval, _ = alphabeta_decider(new_game, max_depth - 1, True, alpha, beta)

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)  # 更新Beta值
            if beta <= alpha:
                break  # Alpha剪枝

        return min_eval, best_move


# 置换表
transposition_table = {}


def alphabeta_decider2(
        game,
        max_depth,
        maximizing_player=True,
        alpha=float("-inf"),
        beta=float("inf")
):
    global count1
    state_key = game.get_state_key()  # 假设你有一个方法来生成当前游戏状态的唯一键

    # 检查置换表
    if state_key in transposition_table:
        return transposition_table[state_key]  # 返回存储的评估值

    # 基线条件：达到最大深度或游戏结束
    if max_depth == 0 or game.is_game_over():
        return evaluate_game_state(game), None

    valid_moves = game.get_valid_moves()  # 获取所有合法的移动
    count1 += len(valid_moves)
    print(count1)

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = alphabeta_decider(new_game, max_depth - 1, False, alpha, beta)

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if alpha >= beta:
                break

                # 存储评估值到置换表
        transposition_table[state_key] = (max_eval, best_move)
        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            eval, _ = alphabeta_decider(new_game, max_depth - 1, True, alpha, beta)

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

                # 存储评估值到置换表
        transposition_table[state_key] = (min_eval, best_move)
        return min_eval, best_move


# 定义历史表
history_table = {}


def update_history_table(move, depth):
    if move in history_table:
        history_table[move] += 1 / depth  # 更新成功率
    else:
        history_table[move] = 1 / depth


def sort_moves_by_history(valid_moves):
    return sorted(valid_moves, key=lambda move: history_table.get(move, 0), reverse=True)


def alphabeta_decider3(
        game,
        max_depth,
        maximizing_player=True,
        alpha=float("-inf"),
        beta=float("inf")
):
    global count1

    # 基线条件：达到最大深度或游戏结束
    if max_depth == 0 or game.is_game_over():
        return evaluate_game_state(game), None

    valid_moves = game.get_valid_moves()  # 获取所有合法的移动
    count1 += len(valid_moves)
    print(count1)

    # 按照历史表排序合法移动
    valid_moves = sort_moves_by_history(valid_moves)

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None

        for move in valid_moves:
            # 复制当前游戏状态
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            # 递归调用，切换为最小化玩家，并传递当前的alpha和beta值
            eval, _ = alphabeta_decider(new_game, max_depth - 1, False, alpha, beta)

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)  # 更新Alpha值
            if alpha >= beta:
                break  # Beta剪枝

        # 更新历史表
        update_history_table(best_move, max_depth)

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None

        for move in valid_moves:
            # 复制当前游戏状态
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            # 递归调用，切换为最大化玩家，并传递当前的alpha和beta值
            eval, _ = alphabeta_decider(new_game, max_depth - 1, True, alpha, beta)

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)  # 更新Beta值
            if beta <= alpha:
                break  # Alpha剪枝

        # 更新历史表
        update_history_table(best_move, max_depth)

        return min_eval, best_move


def mtd_f(game, guess, max_depth):
    """
    MTD(f) algorithm for selecting the best move for the AI player.

    Parameters:
        game (OthelloGame): The current game state.
        guess (float): The initial guess for the evaluation value.
        max_depth (int): The maximum search depth for the Alpha-Beta algorithm.

    Returns:
        tuple: A tuple containing the evaluation value of the best move and the corresponding move (row, col).
    """
    # Initialize alpha and beta bounds
    global best_move
    lower_bound = float("-inf")
    upper_bound = float("inf")

    g = guess

    while lower_bound < upper_bound:
        if g == lower_bound:
            beta = g + 1
        else:
            beta = g

        # Perform a zero-window search using alpha-beta pruning
        g, best_move = alphabeta_decider(game, max_depth, alpha=beta - 1, beta=beta, maximizing_player=True)

        # Update the bounds based on the result
        if g < beta:
            upper_bound = g
        else:
            lower_bound = g

    return g, best_move


def evaluate_game_state(game):
    """
    Evaluates the current game state for the AI player.

    Parameters:
        game (OthelloGame): The current game state.

    Returns:
        float: The evaluation value representing the desirability of the game state for the AI player.
    """
    # Evaluation weights for different factors
    coin_parity_weight = 1.0  # Coin parity (difference in disk count)
    mobility_weight = 2.0  # Mobility (number of valid moves for the current player)
    corner_occupancy_weight = 5.0  # Corner occupancy (number of player disks in the corners)
    stability_weight = 3.0  # Stability (number of stable disks)
    edge_occupancy_weight = 2.5  # Edge occupancy (number of player disks on the edges)

    # Coin parity (difference in disk count)
    player_disk_count = sum(row.count(game.current_player) for row in game.board)
    opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
    coin_parity = player_disk_count - opponent_disk_count

    # Mobility (number of valid moves for the current player)
    player_valid_moves = len(game.get_valid_moves())
    opponent_valid_moves = len(
        OthelloGame(player_mode=-game.current_player).get_valid_moves()
    )
    mobility = player_valid_moves - opponent_valid_moves

    # Corner occupancy (number of player disks in the corners)
    corner_occupancy = sum(
        game.board[i][j] for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)]
    )

    # Stability (number of stable disks)
    stability = calculate_stability(game)

    # Edge occupancy (number of player disks on the edges)
    edge_occupancy = sum(game.board[i][j] for i in [0, 7] for j in range(1, 7)) + sum(
        game.board[i][j] for i in range(1, 7) for j in [0, 7]
    )

    # Combine the factors with the corresponding weights to get the final evaluation value
    evaluation = (
            coin_parity * coin_parity_weight
            + mobility * mobility_weight
            + corner_occupancy * corner_occupancy_weight
            + stability * stability_weight
            + edge_occupancy * edge_occupancy_weight
    )

    return evaluation


def calculate_potential_mobility(game):
    """
    计算当前游戏状态下与对手棋子相邻的空白位置数量（潜在行动力）。

    参数:
        game (OthelloGame): 当前的游戏状态。

    返回:
        int: 潜在行动力的数量。
    """
    opponent = -game.current_player  # 假设当前玩家为 1，对手为 -1
    board = game.board
    potential_moves = set()

    # 定义八个方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    for i in range(8):
        for j in range(8):
            if board[i][j] == opponent:
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < 8 and 0 <= nj < 8:
                        if board[ni][nj] == 0:
                            potential_moves.add((ni, nj))

    return len(potential_moves)


def improved_evaluate_game_state(game):
    """
    Evaluates the current game state for the AI player with improved factors.

    Parameters:
        game (OthelloGame): The current game state.

    Returns:
        float: The evaluation value representing the desirability of the game state for the AI player.
    """
    # Evaluation weights for different factors
    coin_parity_weight = 1.0
    mobility_weight = 2.0
    corner_occupancy_weight = 10.0  # Increased weight for corner control
    stability_weight = 4.0  # Increased weight for stability
    edge_occupancy_weight = 3.0  # Increased weight for edge control
    potential_mobility_weight = 1.5  # New factor for potential mobility

    # Coin parity
    player_disk_count = sum(row.count(game.current_player) for row in game.board)
    opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
    coin_parity = player_disk_count - opponent_disk_count

    # Mobility
    player_valid_moves = len(game.get_valid_moves())
    opponent_valid_moves = len(OthelloGame(player_mode=-game.current_player).get_valid_moves())
    mobility = player_valid_moves - opponent_valid_moves

    # Corner occupancy
    corner_occupancy = sum(game.board[i][j] for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)])

    # Stability
    stability = calculate_stability(game)

    # Edge occupancy
    edge_occupancy = sum(game.board[i][j] for i in [0, 7] for j in range(1, 7)) + sum(
        game.board[i][j] for i in range(1, 7) for j in [0, 7]
    )

    # Potential mobility (number of empty spaces adjacent to opponent's discs)
    potential_mobility = calculate_potential_mobility(game)

    # Combine the factors with the corresponding weights to get the final evaluation value
    evaluation = (
            coin_parity * coin_parity_weight
            + mobility * mobility_weight
            + corner_occupancy * corner_occupancy_weight
            + stability * stability_weight
            + edge_occupancy * edge_occupancy_weight
            + potential_mobility * potential_mobility_weight
    )

    return evaluation


def calculate_stability(game):
    """
    Calculates the stability of the AI player's disks on the board.

    Parameters:
        game (OthelloGame): The current game state.

    Returns:
        int: The number of stable disks for the AI player.
    """

    def neighbors(row, col):
        return [
            (row + dr, col + dc)
            for dr in [-1, 0, 1]
            for dc in [-1, 0, 1]
            if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
        ]

    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    edges = [(i, j) for i in [0, 7] for j in range(1, 7)] + [
        (i, j) for i in range(1, 7) for j in [0, 7]
    ]
    inner_region = [(i, j) for i in range(2, 6) for j in range(2, 6)]
    regions = [corners, edges, inner_region]

    stable_count = 0

    def is_stable_disk(row, col):
        return (
                all(game.board[r][c] == game.current_player for r, c in neighbors(row, col))
                or (row, col) in edges + corners
        )

    for region in regions:
        for row, col in region:
            if game.board[row][col] == game.current_player and is_stable_disk(row, col):
                stable_count += 1

    return stable_count
