package controllers.Astar;

import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Random;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import org.jetbrains.annotations.NotNull;
import tools.ElapsedCpuTimer;
import tools.Vector2d;

/**
 * Agent类是A*算法的实现。
 * 该类扩展自AbstractPlayer，用于在游戏环境中执行智能体的决策，并根据状态观察进行动作选择。
 */
public class Agent extends AbstractPlayer {

    /**
     * 随机数生成器，供智能体使用。
     */
    protected Random randomGenerator;

    /**
     * 观察网格，存储游戏环境中的观察信息。
     */
    protected ArrayList<Observation>[][] grid;

    /**
     * 网格块大小
     */
    protected int block_size; // 格子的大小

    // 存储走过的状态和当前节点的走过的状态
    ArrayList<StateObservation> visitedState = new ArrayList<>();
    ArrayList<StateObservation> targetPastState = new ArrayList<>();

    // 按照f值（评分）比较的优先队列，存储未展开的节点
    Comparator<Node> OrderDistance = Comparator.comparingDouble(x -> x.f);
    PriorityQueue<Node> openState = new PriorityQueue<>(OrderDistance);

    // 存储走过的动作序列以及当前动作的索引
    ArrayList<Types.ACTIONS> actionsSequence = new ArrayList<>();
    int currentActionIndex = -1; // 当前动作的数组下标

    // 记录目标位置和钥匙位置
    Vector2d goalpos; // 目标的位置
    Vector2d keypos; // 钥匙的位置

    // 判断智能体是否找到钥匙
    boolean hasKey = false; // 是否找到钥匙

    // 限制A*算法的搜索深度
    int searchDepth = 32; // 搜索深度限制

    /**
     * 构造函数，接收当前状态观察和时间计时器作为参数。
     *
     * @param so           当前游戏的状态观察
     * @param elapsedTimer 控制器创建的时间计时器
     */
    public Agent(@NotNull StateObservation so, ElapsedCpuTimer elapsedTimer) {
        randomGenerator = new Random(); // 初始化随机数生成器
        grid = so.getObservationGrid(); // 获取观察网格
        block_size = so.getBlockSize(); // 获取格子大小
    }

    /**
     * 检查当前状态是否已经访问过。
     *
     * @param state 当前状态观察
     * @return 如果状态已访问过，返回true；否则返回false。
     */
    boolean duplicateChecking(StateObservation state) {
        for (StateObservation so : visitedState) {
            if (state.equalPosition(so)) {
                return true; // 状态已访问
            }
        }
        return false; // 状态未访问过
    }

    /**
     * 优先级队列中检查当前状态是否存在，并返回对应节点。
     *
     * @param state 当前状态观察
     * @return 如果存在相同状态节点，返回该节点；否则返回null。
     */
    Node priorityDuplicateChecking(StateObservation state) {
        for (Node node : openState) {
            if (state.equalPosition(node.state)) {
                return node; // 返回相同状态的节点
            }
        }
        return null; // 未找到相同状态
    }


    /**
     * 计算从起始状态到当前状态的实际成本g(n)。
     *
     * @param stateObs 当前状态观察
     * @return 从起始状态到当前状态的成本
     */
    double g(StateObservation stateObs) {
        return actionsSequence.size() * 28;  //这个参数可以调整，50效果不是很好，40效果比较好，25以下会在箱子推走后卡住，150以上会在另一个方向卡住
    }

    /**
     * 计算从当前状态到目标状态的启发式成本h(n)。
     *
     * @param stateObs 当前状态观察
     * @param hasKey   指示是否已经找到钥匙
     * @return 当前状态到目标的启发式估计成本
     */
    double h(@NotNull StateObservation stateObs, boolean hasKey) {
        Vector2d playerPos = stateObs.getAvatarPosition(); // 获取精灵位置
        if (hasKey) {
            return Math.abs(goalpos.x - playerPos.x) + Math.abs(goalpos.y - playerPos.y);
        } else {
            double distanceToKey = Math.abs(playerPos.x - keypos.x) + Math.abs(playerPos.y - keypos.y);
            return distanceToKey + Math.abs(goalpos.x - keypos.x) + Math.abs(goalpos.y - keypos.y);
        }
    }

    /**
     * 利用A*算法获取下一步的动作序列。
     *
     * @param stateObs     当前状态观察
     * @param elapsedTimer 时间计时器
     */
    void getAStarActions(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        openState = new PriorityQueue<>(OrderDistance); // 初始化待展开节点的优先队列
        visitedState = new ArrayList<>(targetPastState); // 用历史状态初始化已访问状态
        actionsSequence = new ArrayList<>(); // 初始化已走过的动作序列
        // 创建初始节点，并添加到优先队列
        Node startNode = new Node(stateObs, h(stateObs, hasKey), g(stateObs), actionsSequence, visitedState, hasKey);
        openState.add(startNode);
        // A*搜索
        while (!openState.isEmpty()) { // 当仍有待展开节点时继续搜索
            Node temp = openState.poll(); // 获取评分最优的节点
            actionsSequence = new ArrayList<>(temp.actions); // 克隆动作序列
            targetPastState = new ArrayList<>(temp.pastState); // 克隆当前节点的历史状态
            visitedState.add(temp.state); // 添加当前节点状态到已访问状态列表
            targetPastState.add(temp.state); // 更新当前节点的历史状态
            // 达到搜索深度限制，则退出
            if (actionsSequence.size() == searchDepth) {
                return;
            }
            checkForKey(temp.state, keypos); // 检查是否到达钥匙位置
            // 遍历可用动作并扩展状态
            for (Types.ACTIONS action : temp.state.getAvailableActions()) {
                // 使用新的状态扩展函数
                StateObservation transcript = expandState(temp, action); // 扩展状态
                updateActionsSequence(actionsSequence, action); // 添加动作
                // 使用新的胜利检查函数
                if (checkVictory(transcript)) {
                    return;
                }
                // 检查游戏是否结束或状态是否已访问
                if (isGameOverOrVisited(transcript)) {
                    actionsSequence.remove(actionsSequence.size() - 1); // 移除最后一个动作
                    continue; // 继续尝试下一个动作
                }
                // 处理当前状态在优先队列中的情况
                processNodeInQueue(transcript, actionsSequence); // 调用处理节点函数
            }
        }
    }

    /**
     * 扩展当前状态并处理相关逻辑。
     *
     * @param temp   当前节点
     * @param action 要应用的动作
     * @return 返回扩展后的状态
     */
    StateObservation expandState(@NotNull Node temp, Types.ACTIONS action) {
        StateObservation transcript = temp.state.copy(); // 复制当前状态以进行动作模拟
        transcript.advance(action); // 应用动作
        return transcript; // 返回扩展后的状态
    }

    /**
     * 更新动作序列，添加新的动作。
     *
     * @param actionsSequence 当前的动作序列
     * @param action          被添加的动作
     */
    void updateActionsSequence(@NotNull ArrayList<Types.ACTIONS> actionsSequence, Types.ACTIONS action) {
        actionsSequence.add(action); // 添加动作
    }

    /**
     * 检查当前状态是否胜利。
     *
     * @param state 当前状态观察
     * @return 如果胜利则返回true，否则返回false
     */
    boolean checkVictory(@NotNull StateObservation state) {
        return state.getGameWinner() == Types.WINNER.PLAYER_WINS;
    }

    /**
     * 处理当前状态在优先队列中的情况。
     *
     * @param stCopy          当前状态的副本
     * @param actionsSequence 当前的动作序列
     */
    void processNodeInQueue(StateObservation stCopy, ArrayList<Types.ACTIONS> actionsSequence) {
        Node equalNode = priorityDuplicateChecking(stCopy); // 检查当前状态是否在优先队列中
        if (equalNode != null) { // 找到相同状态的节点
            if (h(stCopy, hasKey) + g(stCopy) < equalNode.f) { // 如果当前路径更优
                // 更新优先队列中的节点
                openState.remove(equalNode);
                openState.add(new Node(stCopy, h(stCopy, hasKey), g(stCopy), actionsSequence, targetPastState, hasKey));
            }
        } else { // 新状态，添加到优先队列
            openState.add(new Node(stCopy, h(stCopy, hasKey), g(stCopy), actionsSequence, targetPastState, hasKey));
        }
        actionsSequence.remove(actionsSequence.size() - 1); // 移除最后一个动作
    }

    /**
     * 检查游戏是否结束或状态是否已访问。
     *
     * @param state 当前状态观察
     * @return 如果游戏结束或状态已访问则返回true，否则返回false
     */
    boolean isGameOverOrVisited(@org.jetbrains.annotations.NotNull StateObservation state) {
        return state.isGameOver() || duplicateChecking(state); // 游戏结束或状态已访问
// 游戏未结束且状态未访问
    }

    /**
     * 检查当前精灵位置是否到达钥匙位置。
     *
     * @param state  当前状态观察
     * @param keypos 钥匙位置
     */
    void checkForKey(StateObservation state, Vector2d keypos) {
        if (!hasKey && state.getAvatarPosition().equals(keypos)) {
            hasKey = true; // 拿到钥匙
        }
    }


    /**
     * 每个游戏步骤调用此方法以请求智能体执行的动作。
     *
     * @param stateObs     当前状态观察
     * @param elapsedTimer 动作返回的计时器
     * @return 当前状态下执行的动作
     */
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        // 初始化目标位置和钥匙位置
        if (actionsSequence.isEmpty()) {
            goalpos = stateObs.getImmovablePositions()[1].get(0).position; // 目标位置
            keypos = stateObs.getMovablePositions()[0].get(0).position; // 钥匙位置
        }

        currentActionIndex++; // 更新当前动作的索引
        // 获取不同类型的位置观察
        ArrayList<Observation>[] npcPositions = stateObs.getNPCPositions();
        ArrayList<Observation>[] fixedPositions = stateObs.getImmovablePositions();
        ArrayList<Observation>[] movingPositions = stateObs.getMovablePositions();
        ArrayList<Observation>[] resourcesPositions = stateObs.getResourcesPositions();
        ArrayList<Observation>[] portalPositions = stateObs.getPortalsPositions();
        grid = stateObs.getObservationGrid(); // 更新观察网格

        // 调试输出
        printDebug(npcPositions, "npc");
        printDebug(fixedPositions, "fix");
        printDebug(movingPositions, "mov");
        printDebug(resourcesPositions, "res");
        printDebug(portalPositions, "por");
        System.out.println();

        // 如果已执行动作集已完成，基于当前状态继续搜索
        if (currentActionIndex == actionsSequence.size()) {
            getAStarActions(stateObs, elapsedTimer); // 更新动作序列
            currentActionIndex = 0; // 重置动作索引
        }
        // 返回当前动作
        return actionsSequence.get(currentActionIndex);
    }

    /**
     * 输出不同类型观察的位置数量。
     *
     * @param positions 观察数组
     * @param str       用于打印的标识符
     */
    private void printDebug(ArrayList<Observation>[] positions, String str) {
        if (positions != null) {
            System.out.print(str + ":" + positions.length + "(");
            for (ArrayList<Observation> position : positions) {
                System.out.print(position.size() + ","); // 打印每个位置类型的观察数量
            }
            System.out.print("); ");
        } else {
            System.out.print(str + ": 0; "); // 如果没有位置，则输出0
        }
    }

    /**
     * 绘制游戏状态信息，仅用于调试。
     *
     * @param g 用于绘制的Graphics设备
     */
    public void draw(Graphics2D g) {
        int half_block = (int) (block_size * 0.5);
        for (int j = 0; j < grid[0].length; ++j) {
            for (int i = 0; i < grid.length; ++i) {
                if (!grid[i][j].isEmpty()) {
                    Observation firstObs = grid[i][j].get(0); // 获取该格子中的第一个观察项
                    int print = firstObs.category; // 获取观察项的类别
                    g.drawString(print + "", i * block_size + half_block, j * block_size + half_block); // 绘制观察信息
                }
            }
        }
    }
}
