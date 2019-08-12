package utilities.iteration.linear;


import utilities.iteration.AbstractIterator;

import java.util.*;

public class LinearIterator<A> extends AbstractIterator<A> {
    protected List<A> values;

    public int getIndex() {
        return index;
    }

    public void setIndex(final int index) {
        this.index = index;
    }

    protected int index = -1;

    public List<A> getValues() {
        return values;
    }

    public void setValues(final List<A> values) {
        this.values = values;
    }

    public LinearIterator(final List<A> values) {
        this.values = new ArrayList<>(values);
    }

    public LinearIterator() {
        this.values = new ArrayList<>();
    }

    public LinearIterator(LinearIterator<A> other) {
        this(other.values);
        index = other.index;
    }

    @Override
    public void remove() {
        values.remove(index);
        index--;
    }

    @Override
    public void add(final A a) {
        values.add(a);
    }

    @Override
    public boolean hasNext() {
        return index + 1 < values.size();
    }

    @Override
    public A next() {
        index++;
        return values.get(index);
    }

    @Override
    public LinearIterator<A> iterator() {
        return new LinearIterator<A>(this);
    }
}
