0. Headers: *objc-rt.hh*
    - rename?
    

1. types: *types.hh*
    - `objc::swap<T>()`
    - `objc::bridge<T,U>()`
    - `objc::bridgeretain<T,U>()`
        - rename to `objc::retain`
    - broken block stuff
    - namespaced types
    - func sigs, `return_sender_t` etc
    - `boolean` and `to_bool`

2. selector wrapper: *selector.hh*
    - `objc::selector`
    - `operator"" _SEL(…)` (originally #8)

3. `objc_msgSend` argument wrapper: *message-args.hh*
    - `objc::arguments` (+ `objc::message`)

4. `traits` namespace: *traits.hh*
    - detail: misc., `test_is_argument_list`, `has_isa`, `has_superclass`
    - detail: SFINAE-friendly `common_type` impl + `is_object_pointer` (at the end)
    - `is_argument_list`, `is_object`, `is_selector`, `is_class`

5. NSObject wrapper template: *object.hh*
    - `objc::object<T>` + `objc::id`

6. `objc_msgSend` high-level API: *message.hh*
    - `objc::msg`

7. <s>`operator"" _SEL(…)`</s>

8. Standard namespace extensions: *namespace-std.hh*
    - `std::swap<…>`
    - `std::hash<…>`

10. IM namespace extensions: *namespace-im.hh*
    - `im::stringify<S>(…)`
    
    