package controllers.Astar;

import core.game.StateObservation;
import ontology.Types;

import java.util.ArrayList;

public class Node {
    public Node(StateObservation state, double h, double g, ArrayList<Types.ACTIONS> actions, ArrayList<StateObservation> pastState, boolean hasKey) {
        this.state = state.copy();
        this.h = h;
        this.g = g;
        this.f = h + g;
        this.actions = new ArrayList<>(actions); // 使用构造函数直接克隆
        this.pastState = new ArrayList<>(pastState); // 使用构造函数直接克隆
        this.hasKey = hasKey;
    } // 初始化

    StateObservation state;
    double g;
    double h;
    double f;
    ArrayList<Types.ACTIONS> actions;
    ArrayList<StateObservation> pastState;
    boolean hasKey;

    public Node parent;
}