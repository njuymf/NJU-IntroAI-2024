package controllers.limitdepthfirst;

import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Stack;
import java.util.PriorityQueue;
import java.util.Comparator;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tools.Vector2d;

/**
 * Created with IntelliJ IDEA.
 * User: ssamot
 * Date: 14/11/13
 * Time: 21:45
 * This is a Java port from Tom Schaul's VGDL - <a href="https://github.com/schaul/py-vgdl">...</a>
 */
public class Agent extends AbstractPlayer {

    /**
     * Random generator for the agent.
     */
    protected Random randomGenerator;

    /**
     * Observation grid.
     */
    protected ArrayList<Observation>[][] grid;

    /**
     * Block size
     */
    protected int block_size;

    /**
     * Store action sequence
     */
    private final List<Types.ACTIONS> actionsSequence = new ArrayList<>();

    /**
     * Store the state of the visited state
     */
    private final List<StateObservation> visitedState = new ArrayList<>();

    /**
     * Whether the path is found
     */
    boolean isWin = false;

    /**
     * Current action index
     */
    private int currentActionIndex = 0; // New index variable
    /**
     * Whether the agent has the key
     */
    boolean hasKey = false;
    /**
     * search depth for single search
     */
    int singleSearchDepth = 0;
    /**
     * maximum search depth
     */
    int searchDepth = 5;

    /**
     * The best score
     */
    double bestScore = Double.MAX_VALUE;
    /**
     * The best action
     */
    List<Types.ACTIONS> bestAction = new ArrayList<>();

    /**
     * Public constructor with state observation and time due.
     * @param so state observation of the current game.
     * @param elapsedTimer Timer for the controller creation.
     */
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer) {
        randomGenerator = new Random();
        grid = so.getObservationGrid();
        block_size = so.getBlockSize();
    }

    /**
     * Compare the current state with the visited state
     * @param state the current state
     * @return whether the current state is visited
     */
    boolean duplicateChecking(StateObservation state) {
        return visitedState.stream().anyMatch(state::equalPosition);
    }

    /**
     * Iterative Depth-First Search using an explicit stack
     * @param initialState the initial state observation
     * @param elapsedTimer timer for the search
     * @return true if a winning path is found, false otherwise
     */
    boolean getDepthFirstActionsIterative(StateObservation initialState, ElapsedCpuTimer elapsedTimer) {
        long startTime = System.currentTimeMillis();
        Vector2d goalpos = initialState.getImmovablePositions()[1].get(0).position; //目标的坐标
        Vector2d keypos = initialState.getMovablePositions()[0].get(0).position; //钥匙的坐标

        PriorityQueue<Node> queue = new PriorityQueue<>(Comparator.comparingDouble(n -> n.distance));
        queue.add(new Node(initialState, new ArrayList<>(), 0, distance(initialState, goalpos, keypos)));


        while (!queue.isEmpty()) {
            printQueue(queue);

            Node currentNode = queue.poll();
            StateObservation currentState = null;
            if (currentNode != null) {
                currentState = currentNode.state;
            }else {
                System.out.println("Current node is null");
            }

            List<Types.ACTIONS> currentActions = null;
            if (currentNode != null) {
                currentActions = currentNode.actions;
            }else {
                System.out.println("Current actions is null");
            }

            int currentDepth = 0;
            if (currentNode != null) {
                currentDepth = currentNode.depth;
            }else{
                System.out.println("Current depth is null");
            }

            if (checkWinCondition(currentState)) {
                // 设置胜利状态的极值评分
                bestScore = Double.NEGATIVE_INFINITY;
                if (currentActions != null) {
                    bestAction = new ArrayList<>(currentActions);
                }else {
                    System.out.println("severe error: currentActions is null in final state");
                }
                actionsSequence.clear();
                actionsSequence.addAll(bestAction);
                currentActionIndex = 0;

                long endTime = System.currentTimeMillis();
                long totalTime = endTime - startTime;
                System.out.println("Total time taken for limitDFS: " + totalTime + " ms");
                return true;
            }

            if (processState(currentState)) {
                visitedState.add(currentState);

                for (Types.ACTIONS action : currentState.getAvailableActions()) {
                    StateObservation nextState = currentState.copy();
                    nextState.advance(action);

                    List<Types.ACTIONS> newActions = new ArrayList<>(currentActions);
                    newActions.add(action);

                    // 仅当未找到胜利路径时才更新评分
                    if (currentDepth + 1 == searchDepth && bestScore != Double.NEGATIVE_INFINITY) {
                        double score = -50 * (searchDepth - currentDepth);
                        if (score < bestScore) {
                            bestScore = score;
                            bestAction = new ArrayList<>(newActions);
                        }
                    }
                    queue.add(new Node(nextState, newActions, currentDepth + 1, distance(nextState, goalpos, keypos)));
                }
            }
        }

        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Total time taken for limitDFS: " + totalTime + " ms");
        return false;
    }
    /**
     * Helper class to store state and action sequence
     */
    private static class Node {
        StateObservation state;
        List<Types.ACTIONS> actions;
        int depth;

        double distance;

        Node(StateObservation state, List<Types.ACTIONS> actions,int depth,double distance) {
            this.state = state;
            this.actions = actions;
            this.depth = depth;
            this.distance = distance;
        }
    }

    /**
     * Check if the current state results in a win
     * @param reproduction the current state after action
     * @return true if the game is won, false otherwise
     */
    private boolean checkWinCondition(StateObservation reproduction) {
        return reproduction.getGameWinner() == Types.WINNER.PLAYER_WINS;
    }

    /**
     * Process the state to check if it has been visited or if the game is over
     * @param reproduction the current state after action
     * @return true if the state is valid for further exploration, false otherwise
     */
    private boolean processState(StateObservation reproduction) {
        // Check if the current state is visited or the game is over
        return !duplicateChecking(reproduction) && !reproduction.isGameOver(); // State is valid for further exploration
    }

    /**
     * Picks an action. This function is called every game step to request an
     * action from the player.
     * @param stateObs Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
        ArrayList<Observation>[] npcPositions = stateObs.getNPCPositions();
        ArrayList<Observation>[] fixedPositions = stateObs.getImmovablePositions();
        ArrayList<Observation>[] movingPositions = stateObs.getMovablePositions();
        ArrayList<Observation>[] resourcesPositions = stateObs.getResourcesPositions();
        ArrayList<Observation>[] portalPositions = stateObs.getPortalsPositions();
        grid = stateObs.getObservationGrid();

        printDebug(npcPositions,"npc");
        printDebug(fixedPositions,"fix");
        printDebug(movingPositions,"mov");
        printDebug(resourcesPositions,"res");
        printDebug(portalPositions,"por");
        System.out.println();

        if(!isWin) {
            isWin = getDepthFirstActionsIterative(stateObs, elapsedTimer);
        }

        // Use a temporary variable to check bounds
        if (currentActionIndex < actionsSequence.size()) {
            Types.ACTIONS action = actionsSequence.get(currentActionIndex);
            currentActionIndex++;
            return action;
        }
        return null; // No action available
    }

    /**
     * Prints the number of different types of sprites available in the "positions" array.
     * Between brackets, the number of observations of each type.
     * @param positions array with observations.
     * @param str identifier to print
     */
    private void printDebug(ArrayList<Observation>[] positions, String str) {
        if (positions != null) {
            System.out.print(str + ":" + positions.length + "(");
            for (ArrayList<Observation> position : positions) {
                System.out.print(position.size() + ",");
            }
            System.out.print("); ");
        } else {
            System.out.print(str + ": 0; ");
        }
    }

    /**
     * Gets the player the control to draw something on the screen.
     * It can be used for debug purposes.
     * @param g Graphics device to draw to.
     */
    public void draw(Graphics2D g) {
        int half_block = (int) (block_size * 0.5);
        for (int j = 0; j < grid[0].length; ++j) {
            for (int i = 0; i < grid.length; ++i) {
                if (!grid[i][j].isEmpty()) {
                    Observation firstObs = grid[i][j].get(0);
                    int print = firstObs.category;
                    g.drawString(print + "", i * block_size + half_block, j * block_size + half_block);
                }
            }
        }
    }

    /**
     * Prints the contents of the stack
     * @param queue the queue to print
     */
    private void printQueue(PriorityQueue<Node> queue) {
        System.out.println("Current queue contents:");

        // 创建一个临时列表来存储队列中的节点
        List<Node> tempList = new ArrayList<>(queue);

        // 遍历临时列表
        for (Node node : tempList) {
            StringBuilder actionsOutput = new StringBuilder();
            for (Types.ACTIONS action : node.actions) {
                switch (action) {
                    case ACTION_UP:
                        actionsOutput.append("↑ ");
                        break;
                    case ACTION_DOWN:
                        actionsOutput.append("↓ ");
                        break;
                    case ACTION_LEFT:
                        actionsOutput.append("← ");
                        break;
                    case ACTION_RIGHT:
                        actionsOutput.append("→ ");
                        break;
                    default:
                        actionsOutput.append(action).append(" ");
                        break;
                }
            }
            System.out.println("State: " + node.state + ", Actions: [" + actionsOutput.toString().trim() + "]");
        }
    }
    double distance(StateObservation stateObs, Vector2d goalpos, Vector2d keypos) {
        Vector2d playerPos = stateObs.getAvatarPosition(); // 精灵的位置


        // if the player has the key
        if (hasKey) {
            return Math.abs(goalpos.x - playerPos.x) + Math.abs(goalpos.y - playerPos.y);
        }

        // if the player has visited the key
        boolean hasVisitedKey = visitedState.stream()
                .anyMatch(so -> so.getAvatarPosition().equals(keypos));

        //if the player has visited the key
        if (hasVisitedKey) {
            return Math.abs(goalpos.x - playerPos.x) + Math.abs(goalpos.y - playerPos.y);
        }

        // if the player has not visited the key
        return Math.abs(playerPos.x - keypos.x) + Math.abs(playerPos.y - keypos.y) + Math.abs(goalpos.x - keypos.x) + Math.abs(goalpos.y - keypos.y);
    }
}

