package controllers.depthfirst;

import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Stack;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import tools.ElapsedCpuTimer;

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
        long startTime = System.currentTimeMillis(); // 记录开始时间

        // Define a stack to hold pairs of state and the corresponding action sequence
        Stack<Node> stack = new Stack<>();
        stack.push(new Node(initialState, new ArrayList<>()));

        while (!stack.isEmpty()) {
            printStack(stack);

            Node currentNode = stack.pop();
            StateObservation currentState = currentNode.state;
            List<Types.ACTIONS> currentActions = currentNode.actions;

            // Check for win condition
            if (checkWinCondition(currentState)) {
                actionsSequence.clear();
                actionsSequence.addAll(currentActions);
                long endTime = System.currentTimeMillis(); // 记录结束时间
                long totalTime = endTime - startTime; // 计算总耗时
                System.out.println("Total time taken for DFS: " + totalTime + " ms"); // 输出总耗时
                return true;
            }

            // If state is valid for exploration
            if (processState(currentState)) {
                // Mark the state as visited
                visitedState.add(currentState);

                // Iterate through available actions
                for (Types.ACTIONS action : currentState.getAvailableActions()) {
                    StateObservation nextState = currentState.copy();
                    nextState.advance(action);

                    List<Types.ACTIONS> newActions = new ArrayList<>(currentActions);
                    newActions.add(action);

                    stack.push(new Node(nextState, newActions));
                }
            }
        }

        long endTime = System.currentTimeMillis(); // 记录结束时间
        long totalTime = endTime - startTime; // 计算总耗时
        System.out.println("Total time taken for DFS: " + totalTime + " ms"); // 输出总耗时
        return false; // No path found
    }

    /**
     * Helper class to store state and action sequence
     */
    private static class Node {
        StateObservation state;
        List<Types.ACTIONS> actions;

        Node(StateObservation state, List<Types.ACTIONS> actions) {
            this.state = state;
            this.actions = actions;
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
     * Print the contents of the stack
     * @param stack the stack to print
     */
    private void printStack(Stack<Node> stack) {
        System.out.println("Current stack contents:");
        for (Node node : stack) {
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
}