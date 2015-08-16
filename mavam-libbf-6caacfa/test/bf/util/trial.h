#ifndef UTIL_TRIAL_H
#define UTIL_TRIAL_H

#include <cassert>
#include "util/error.h"

namespace util {

/// Represents the result of a computation which can either complete
/// successfully with an instance of type `T` or fail with an ::error.
///
/// @tparam The type of the result.
template <typename T>
class trial
{
public:
  /// Constructs a trial by forwarding arguments to the underlying type.
  /// @param args The arguments to pass to `T`'s constructor.
  /// @post The trial is *engaged*, i.e., `*this == true`.
  trial(T x)
    : engaged_{true}
  {
    new (&value_) T(std::move(x));
  }

  /// Constructs a trial from an error.
  /// @param e The error.
  /// @post The trial is *disengaged*, i.e., `*this == false`.
  trial(error e)
    : engaged_{false}
  {
    new (&error_) error{std::move(e)};
  }

  /// Copy-constructs a trial.
  /// @param other The other trial.
  trial(trial const& other)
  {
    construct(other);
  }

  /// Move-constructs a trial.
  /// @param other The other trial.
  trial(trial&& other)
  {
    construct(std::move(other));
  }

  ~trial()
  {
    destroy();
  }

  /// Assigns another trial to this instance.
  /// @param other The RHS of the assignment.
  /// @returns A reference to `*this`.
  trial& operator=(trial other)
  {
    construct(std::move(other));
    return *this;
  }

  /// Assigns a trial by constructing an instance of `T` from the RHS
  /// expression.
  ///
  /// @param args The arguments to forward to `T`'s constructor.
  /// @returns A reference to `*this`.
  trial& operator=(T x)
  {
    destroy();
    engaged_ = true;
    new (&value_) T(std::move(x));
    return *this;
  }

  /// Assigns an error to this trial instance.
  /// @param e The error.
  /// @returns A reference to `*this`.
  /// @post The trial is *disengaged*, i.e., `*this == false`.
  trial& operator=(error e)
  {
    destroy();
    engaged_ = false;
    new (&value_) error{std::move(e)};
    return *this;
  }

  /// Checks whether the trial is engaged.
  /// @returns `true` iff the trial is engaged.
  explicit operator bool() const
  {
    return engaged_;
  }

  /// Shorthand for ::value.
  T& operator*()
  {
    return value();
  }

  /// Shorthand for ::value.
  T const& operator*() const
  {
    return value();
  }

  /// Shorthand for ::value.
  T* operator->()
  {
    return &value();
  }

  /// Shorthand for ::value.
  T const* operator->() const
  {
    return &value();
  }

  /// Retrieves the value of the trial.
  /// @returns A mutable reference to the contained value.
  /// @pre `*this == true`.
  T& value()
  {
    assert(engaged_);
    return value_;
  }

  /// Retrieves the value of the trial.
  /// @returns The contained value.
  /// @pre `*this == true`.
  T const& value() const
  {
    assert(engaged_);
    return value_;
  }

  /// Retrieves the error of the trial.
  /// @returns The contained error.
  /// @pre `*this == false`.
  error const& failure() const
  {
    assert(! engaged_);
    return error_;
  }

private:
  void construct(trial const& other)
  {
    if (other.engaged_)
    {
      engaged_ = true;
      new (&value_) T(other.value_);
    }
    else
    {
      engaged_ = false;
      new (&error_) error{other.error_};
    }
  }

  void construct(trial&& other)
  {
    if (other.engaged_)
    {
      engaged_ = true;
      new (&value_) T(std::move(other.value_));
    }
    else
    {
      engaged_ = false;
      new (&error_) error{std::move(other.error_)};
    }
  }

  void destroy()
  {
    if (engaged_)
      value_.~T();
    else
      error_.~error();
  }

  union
  {
    T value_;
    error error_;
  };

  bool engaged_;
};

/// An empty struct that represents a `void` ::trial. The pattern
/// `trial<nothing>` shall be used for functions that may generate an error but
/// would otherwise return `void`.
struct nothing { };

static constexpr auto nil = nothing{};

} // namespace util

#endif
