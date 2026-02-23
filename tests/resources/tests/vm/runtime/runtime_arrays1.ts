function main() {
    // 1. Array creation and mutation
    var arr1 = array_create(5);
    array_set(arr1, 0, 10.0);
    array_set(arr1, 1, 20.0);
    array_set(arr1, 2, 30.0);
    var first_element = array_get(arr1, 0);

    // 2. Orchestration verbs (Opaque composition)
    var smoothed = rolling_mean(arr1, 2);

    // Test that rolling mean function produced expected value at index 1
    var smoothed_val = array_get(smoothed, 1);

    // Clean up
    array_free(arr1);
    array_free(smoothed);

    return smoothed_val; // Should return (10+20)/2 = 15.0
}
